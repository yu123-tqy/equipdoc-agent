from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from ..config import Settings
from ..rag import KnowledgeRetriever
from .agentic_tools import execute_agentic_tool
from .knowledge_answer import (
    build_grounded_synthesis_messages,
    build_grounded_synthesis_retry_messages,
    build_ranked_evidence_candidates,
    render_safe_fallback,
    render_structured_evidence_answer,
    render_tool_observation_section,
    select_evidence_for_question,
    should_retry_grounded_synthesis,
    validate_answer_citations,
    validate_grounded_draft,
)
from .planning import (
    PlanningValidationError,
    build_intent_plan_messages,
    build_intent_plan_retry_messages,
    build_observation_messages,
    fallback_plan,
    parse_and_validate_plan,
    parse_observation_decision,
)
from .policy import normalize_review_decision
from .safety import assess_high_risk_question


class AgenticState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    signal_path: str
    review_result: str
    current_plan: dict[str, Any]
    pending_tool_call: dict[str, Any]
    tool_observations: list[dict[str, Any]]
    tool_step_count: int
    session_memory: dict[str, Any]
    planning_metadata: dict[str, Any]
    observation_metadata: dict[str, Any]
    answer_metadata: dict[str, Any]
    safety_decision: dict[str, str]
    retrieval_hits: list[dict[str, Any]]


def _last_human_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _response_text(response: Any) -> str:
    return str(getattr(response, "content", response))


def _review_call_payload(call: dict[str, Any], signal_path: str | None) -> dict[str, Any]:
    arguments = dict(call.get("arguments") or {})
    arguments.pop("signal_path", None)
    if signal_path and call.get("tool") in {"diagnose_bearing", "inspect_signal"}:
        arguments["signal_file"] = Path(signal_path).name
    return {"name": call.get("tool"), "args": arguments}


def _new_turn_memory(state: AgenticState, user_text: str) -> dict[str, Any]:
    memory = deepcopy(state.get("session_memory") or {})
    if any(term in user_text for term in ("忘记当前任务", "重置会话", "清除上下文")):
        memory = {}
    signal_path = state.get("signal_path") or ""
    signal_file = Path(signal_path).name if signal_path else memory.get("signal_file", "")
    previous_file = memory.get("signal_file")
    if signal_file and previous_file and signal_file != previous_file:
        for key in (
            "last_diagnosis",
            "last_search_query",
            "last_evidence",
            "pending_clarification",
            "completed_tools",
            "attempted_tools",
            "failed_tools",
        ):
            memory.pop(key, None)
    if signal_file:
        memory["signal_file"] = signal_file
    memory["completed_tools"] = []
    memory["attempted_tools"] = []
    memory["failed_tools"] = []
    return memory


def _completed_step_ids(observations: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["_step_id"])
        for item in observations
        if item.get("_step_id") and item.get("status") not in {"error", "skipped"}
    }


def _ready_plan_steps(state: AgenticState) -> list[dict[str, Any]]:
    observations = state.get("tool_observations") or []
    completed = _completed_step_ids(observations)
    steps = list((state.get("current_plan") or {}).get("plan") or [])
    return [
        step
        for step in steps
        if step.get("step_id") not in completed
        and set(step.get("depends_on") or []).issubset(completed)
    ]


def _permitted_observation_tools(state: AgenticState) -> set[str]:
    permitted = {str(step["tool"]) for step in _ready_plan_steps(state)}
    observations = state.get("tool_observations") or []
    completed_tools = {str(item.get("_tool_name")) for item in observations}
    last_tool = str(observations[-1].get("_tool_name")) if observations else ""
    if (
        last_tool in {"diagnose_bearing", "inspect_signal"}
        and "search_maintenance_knowledge" not in completed_tools
    ):
        permitted.add("search_maintenance_knowledge")
    return permitted


def _fault_filter_value(fault_type: str) -> str | None:
    if "外圈" in fault_type:
        return "outer_race"
    if "内圈" in fault_type:
        return "inner_race"
    if "滚动体" in fault_type:
        return "ball"
    return None


def _fallback_observation_decision(
    state: AgenticState,
    permitted_tools: set[str],
    *,
    reason: str,
) -> dict[str, Any]:
    observations = state.get("tool_observations") or []
    last = observations[-1] if observations else {}
    if (
        last.get("status") != "error"
        and last.get("_tool_name") == "diagnose_bearing"
        and "search_maintenance_knowledge" in permitted_tools
    ):
        fault_type = str(last.get("fault_type", "轴承故障"))
        arguments: dict[str, Any] = {
            "query": f"{fault_type} 故障机理 振动特征 现场复核",
            "equipment": "bearing",
            "top_k": 5,
        }
        fault_filter = _fault_filter_value(fault_type)
        if fault_filter:
            arguments["fault_type"] = fault_filter
        return {
            "action": "call_tool",
            "tool": "search_maintenance_knowledge",
            "arguments": arguments,
            "reason": reason,
            "clarification_question": "",
            "validation": {"source": "deterministic_fallback"},
        }
    successful = [item for item in observations if item.get("status") != "error"]
    if not successful:
        error = str(last.get("error", "工具没有返回可用结果"))
        return {
            "action": "clarify",
            "tool": None,
            "arguments": {},
            "reason": reason,
            "clarification_question": f"工具执行未得到可用结果：{error}。请检查输入后重试。",
            "validation": {"source": "deterministic_fallback"},
        }
    return {
        "action": "answer",
        "tool": None,
        "arguments": {},
        "reason": reason,
        "clarification_question": "",
        "validation": {"source": "deterministic_fallback"},
    }


def _merge_memory_after_tool(
    memory: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(memory)
    tool_name = str(observation.get("_tool_name", ""))
    status = str(observation.get("status", ""))
    attempted = list(updated.get("attempted_tools") or [])
    if tool_name and tool_name not in attempted:
        attempted.append(tool_name)
    updated["attempted_tools"] = attempted
    if status in {"error", "skipped"}:
        failed = list(updated.get("failed_tools") or [])
        if tool_name and tool_name not in failed:
            failed.append(tool_name)
        updated["failed_tools"] = failed
        return updated
    completed = list(updated.get("completed_tools") or [])
    if tool_name and tool_name not in completed:
        completed.append(tool_name)
    updated["completed_tools"] = completed
    if observation.get("signal_file"):
        updated["signal_file"] = observation["signal_file"]
    if tool_name == "diagnose_bearing" and observation.get("status") != "error":
        updated["current_equipment"] = "bearing"
        updated["last_diagnosis"] = {
            "fault_type": observation.get("fault_type"),
            "confidence": observation.get("confidence"),
            "warning": observation.get("warning"),
        }
    if tool_name == "search_maintenance_knowledge":
        updated["last_search_query"] = observation.get("query", "")
        updated["last_evidence"] = [
            {
                "citation": hit.get("citation"),
                "title": hit.get("title"),
                "text": hit.get("text"),
            }
            for hit in list(observation.get("hits") or [])[:5]
        ]
    return updated


def _collect_search_hits(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for round_index, observation in enumerate(observations, start=1):
        if observation.get("_tool_name") != "search_maintenance_knowledge":
            continue
        for hit in observation.get("hits") or []:
            chunk_id = str(hit.get("chunk_id", ""))
            if chunk_id:
                enriched = {
                    **hit,
                    "retrieval_rounds": [round_index],
                    "focused_match": bool(hit.get("focused_match")) or round_index > 1,
                }
                if chunk_id not in unique:
                    unique[chunk_id] = enriched
                    continue
                existing = unique[chunk_id]
                existing["retrieval_rounds"] = sorted(
                    set(existing.get("retrieval_rounds") or []).union({round_index})
                )
                existing["focused_match"] = bool(existing.get("focused_match")) or bool(
                    enriched.get("focused_match")
                )
                for score_key in ("rrf_score", "lexical_score", "dense_score"):
                    values = [
                        value
                        for value in (existing.get(score_key), enriched.get(score_key))
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    ]
                    if values:
                        existing[score_key] = max(values)
    return list(unique.values())


def _has_usable_tool_observation(observations: list[dict[str, Any]]) -> bool:
    for observation in observations:
        if observation.get("status") in {"error", "skipped"}:
            continue
        if observation.get("_tool_name") == "search_maintenance_knowledge":
            if observation.get("hits"):
                return True
            continue
        return True
    return False


def _render_safety_decision(decision: dict[str, str]) -> str:
    return (
        f"## 安全边界（{decision.get('policy_id', 'agentic_safety')}）\n\n"
        f"{decision.get('message', '该请求超出当前 Agent 的安全能力边界。')}\n\n"
        "系统未执行任何工具；请由具备权限的工程人员结合现场信息处理。"
    )


def build_agentic_graph(
    settings: Settings,
    *,
    llm: Any | None = None,
    retriever: KnowledgeRetriever | None = None,
    checkpointer=None,
):
    if settings.demo_mode:
        raise ValueError("Agentic graph is available only when demo_mode is false.")
    model = llm or ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        temperature=0,
    )
    retriever_holder: dict[str, KnowledgeRetriever | None] = {}
    if retriever is not None:
        retriever_holder["retriever"] = retriever

    def get_retriever() -> KnowledgeRetriever | None:
        if "retriever" in retriever_holder:
            return retriever_holder["retriever"]
        if not settings.rag_enabled or not settings.rag_chunks_path.exists():
            retriever_holder["retriever"] = None
        else:
            retriever_holder["retriever"] = KnowledgeRetriever(settings)
        return retriever_holder["retriever"]

    def policy_gate_node(state: AgenticState) -> dict[str, Any]:
        user_text = _last_human_text(state.get("messages", []))
        decision = assess_high_risk_question(user_text)
        if decision is None:
            return {"safety_decision": {}}
        return {
            "safety_decision": {
                "policy_id": decision.policy_id,
                "message": decision.message,
            }
        }

    def route_after_policy(state: AgenticState):
        return "safety_response" if state.get("safety_decision") else "planner"

    def safety_response_node(state: AgenticState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(content=_render_safety_decision(state.get("safety_decision") or {}))
            ]
        }

    def planner_node(state: AgenticState) -> dict[str, Any]:
        user_text = _last_human_text(state.get("messages", []))
        signal_path = state.get("signal_path") or ""
        previous_signal_file = str((state.get("session_memory") or {}).get("signal_file", ""))
        memory = _new_turn_memory(state, user_text)
        replacement_requested = any(
            term in user_text for term in ("换一个文件", "更换文件", "换份信号")
        )
        if (
            replacement_requested
            and previous_signal_file
            and Path(signal_path).name == previous_signal_file
        ):
            signal_path = ""
            for key in (
                "signal_file",
                "last_diagnosis",
                "last_search_query",
                "last_evidence",
                "completed_tools",
                "attempted_tools",
                "failed_tools",
            ):
                memory.pop(key, None)
        messages = build_intent_plan_messages(
            user_text,
            memory=memory,
            has_signal=bool(signal_path),
            max_steps=settings.agent_max_steps,
        )
        llm_calls: list[dict[str, Any]] = []
        call_started = time.perf_counter()
        first_response = model.invoke(messages)
        llm_calls.append(
            {
                "stage": "planning_first_pass",
                "latency_seconds": time.perf_counter() - call_started,
            }
        )
        attempts = 1
        errors: list[str] = []
        try:
            plan = parse_and_validate_plan(
                _response_text(first_response),
                max_steps=settings.agent_max_steps,
                has_signal=bool(signal_path),
                user_text=user_text,
            )
            generation_path = "first_pass"
        except PlanningValidationError as first_error:
            errors.append(str(first_error))
            retry_messages = build_intent_plan_retry_messages(
                user_text,
                rejected_output=_response_text(first_response),
                validation_error=str(first_error),
                memory=memory,
                has_signal=bool(signal_path),
                max_steps=settings.agent_max_steps,
            )
            call_started = time.perf_counter()
            second_response = model.invoke(retry_messages)
            llm_calls.append(
                {
                    "stage": "planning_retry",
                    "latency_seconds": time.perf_counter() - call_started,
                }
            )
            attempts = 2
            try:
                plan = parse_and_validate_plan(
                    _response_text(second_response),
                    max_steps=settings.agent_max_steps,
                    has_signal=bool(signal_path),
                    user_text=user_text,
                )
                generation_path = "retry"
            except PlanningValidationError as second_error:
                errors.append(str(second_error))
                plan = fallback_plan(
                    user_text,
                    has_signal=bool(signal_path),
                    max_steps=settings.agent_max_steps,
                )
                generation_path = "deterministic_fallback"
        if plan["intent"] != "clarification":
            memory["pending_clarification"] = ""
        return {
            "current_plan": plan,
            "pending_tool_call": {},
            "tool_observations": [],
            "tool_step_count": 0,
            "session_memory": memory,
            "planning_metadata": {
                "generation_path": generation_path,
                "attempts": attempts,
                "validation_errors": errors,
                "llm_calls": llm_calls,
            },
            "observation_metadata": {},
            "answer_metadata": {},
            "review_result": "",
            "signal_path": signal_path,
            "retrieval_hits": [],
        }

    def route_after_planner(state: AgenticState):
        intent = (state.get("current_plan") or {}).get("intent")
        if intent == "clarification":
            return "clarification"
        if intent == "safety_boundary":
            return "safety_response"
        return "prepare_tool"

    def prepare_tool_node(state: AgenticState) -> dict[str, Any]:
        ready = _ready_plan_steps(state)
        if not ready:
            return {"pending_tool_call": {}}
        step = ready[0]
        return {
            "pending_tool_call": {
                "step_id": step["step_id"],
                "tool": step["tool"],
                "arguments": dict(step.get("arguments") or {}),
                "source": "initial_plan",
            }
        }

    def route_pending_tool(state: AgenticState):
        call = state.get("pending_tool_call") or {}
        if not call:
            return "synthesizer"
        return "review" if call.get("tool") == "diagnose_bearing" else "tools"

    def review_node(state: AgenticState) -> dict[str, Any]:
        call = state.get("pending_tool_call") or {}
        decision = interrupt(
            {
                "type": "tool_review",
                "requested_tools": [_review_call_payload(call, state.get("signal_path"))],
                "notice": ("Approve runs the sandboxed diagnostic tool; Reject cancels the call."),
            }
        )
        return {"review_result": str(decision)}

    def after_review(state: AgenticState):
        try:
            decision = normalize_review_decision(state.get("review_result", ""))
        except ValueError:
            return "cancel"
        return "tools" if decision == "approve" else "cancel"

    def tool_node(state: AgenticState) -> dict[str, Any]:
        call = state.get("pending_tool_call") or {}
        step_count = int(state.get("tool_step_count") or 0)
        if step_count >= settings.agent_max_steps:
            observation = {
                "_tool_name": str(call.get("tool", "unknown")),
                "_step_id": str(call.get("step_id", "")),
                "status": "error",
                "error": "Agent tool step limit reached.",
            }
        else:
            observation = execute_agentic_tool(
                str(call.get("tool", "")),
                dict(call.get("arguments") or {}),
                signal_path=state.get("signal_path"),
                settings=settings,
                retriever=get_retriever(),
            )
            observation["_step_id"] = str(call.get("step_id") or f"O{step_count + 1}")
            observation["_source"] = str(call.get("source", "unknown"))
            step_count += 1
        observations = list(state.get("tool_observations") or []) + [observation]
        memory = _merge_memory_after_tool(
            state.get("session_memory") or {},
            observation,
        )
        tool_message = ToolMessage(
            content=json.dumps(observation, ensure_ascii=False),
            tool_call_id=f"agentic_{uuid4().hex}",
        )
        return {
            "messages": [tool_message],
            "tool_observations": observations,
            "tool_step_count": step_count,
            "session_memory": memory,
            "pending_tool_call": {},
        }

    def route_after_tool(state: AgenticState):
        observations = state.get("tool_observations") or []
        last = observations[-1] if observations else {}
        if last.get("status") in {"error", "skipped"}:
            return "observer"
        if _ready_plan_steps(state):
            return "prepare_tool"
        if last.get("_tool_name") == "diagnose_bearing":
            completed_tools = {str(item.get("_tool_name")) for item in observations}
            if "search_maintenance_knowledge" not in completed_tools:
                return "diagnosis_evidence"
        return "synthesizer"

    def diagnosis_evidence_node(state: AgenticState) -> dict[str, Any]:
        decision = _fallback_observation_decision(
            state,
            {"search_maintenance_knowledge"},
            reason="诊断工具成功后按故障类别补充可回查知识证据。",
        )
        update: dict[str, Any] = {
            "observation_metadata": {
                "generation_path": "deterministic_diagnosis_followup",
                "validation_error": "",
                "decision": decision,
                "llm_calls": [],
            }
        }
        if decision.get("action") == "call_tool":
            update["pending_tool_call"] = {
                "step_id": f"O{int(state.get('tool_step_count') or 0) + 1}",
                "tool": decision["tool"],
                "arguments": decision["arguments"],
                "source": "deterministic_diagnosis_followup",
            }
        return update

    def observer_node(state: AgenticState) -> dict[str, Any]:
        remaining_steps = max(
            0,
            settings.agent_max_steps - int(state.get("tool_step_count") or 0),
        )
        permitted = _permitted_observation_tools(state)
        if remaining_steps == 0:
            decision = _fallback_observation_decision(
                state,
                set(),
                reason="已达到最大工具步数，停止继续调用。",
            )
            return {
                "observation_metadata": {
                    "generation_path": "max_steps_stop",
                    "decision": decision,
                    "llm_calls": [],
                }
            }
        messages = build_observation_messages(
            _last_human_text(state.get("messages", [])),
            current_plan=state.get("current_plan") or {},
            observations=state.get("tool_observations") or [],
            permitted_tools=permitted,
            remaining_steps=remaining_steps,
            memory=state.get("session_memory") or {},
        )
        call_started = time.perf_counter()
        response = model.invoke(messages)
        llm_call = {
            "stage": "observation_decision",
            "latency_seconds": time.perf_counter() - call_started,
        }
        try:
            decision = parse_observation_decision(
                _response_text(response),
                permitted_tools=permitted,
                remaining_steps=remaining_steps,
                has_signal=bool(state.get("signal_path")),
                has_usable_observation=_has_usable_tool_observation(
                    state.get("tool_observations") or []
                ),
            )
            generation_path = "model"
            validation_error = ""
        except PlanningValidationError as exc:
            validation_error = str(exc)
            decision = _fallback_observation_decision(
                state,
                permitted,
                reason=f"观察决策未通过校验：{exc}",
            )
            generation_path = "deterministic_fallback"

        update: dict[str, Any] = {
            "observation_metadata": {
                "generation_path": generation_path,
                "validation_error": validation_error,
                "decision": decision,
                "llm_calls": [llm_call],
            }
        }
        if decision["action"] == "call_tool":
            ready = [
                step for step in _ready_plan_steps(state) if step.get("tool") == decision["tool"]
            ]
            step_id = (
                str(ready[0]["step_id"])
                if ready
                else f"O{int(state.get('tool_step_count') or 0) + 1}"
            )
            update["pending_tool_call"] = {
                "step_id": step_id,
                "tool": decision["tool"],
                "arguments": decision["arguments"],
                "source": "observation_decision",
            }
        elif decision["action"] == "clarify":
            memory = deepcopy(state.get("session_memory") or {})
            memory["pending_clarification"] = decision["clarification_question"]
            update["session_memory"] = memory
        return update

    def route_after_observer(state: AgenticState):
        decision = (state.get("observation_metadata") or {}).get("decision") or {}
        action = decision.get("action", "answer")
        if action == "clarify":
            return "clarification"
        if action == "call_tool":
            call = state.get("pending_tool_call") or {}
            return "review" if call.get("tool") == "diagnose_bearing" else "tools"
        return "synthesizer"

    def clarification_node(state: AgenticState) -> dict[str, Any]:
        plan = state.get("current_plan") or {}
        observation_decision = (state.get("observation_metadata") or {}).get("decision") or {}
        question = (
            observation_decision.get("clarification_question")
            or plan.get("clarification_question")
            or "当前信息不足，请补充设备、信号或现场现象后再继续。"
        )
        memory = deepcopy(state.get("session_memory") or {})
        memory["pending_clarification"] = question
        return {
            "messages": [AIMessage(content=f"## 需要补充信息\n\n{question}")],
            "session_memory": memory,
        }

    def cancel_node(_: AgenticState) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="已根据人工审核取消本次诊断工具调用。")],
            "pending_tool_call": {},
        }

    def synthesizer_node(state: AgenticState) -> dict[str, Any]:
        observations = state.get("tool_observations") or []
        user_text = _last_human_text(state.get("messages", []))
        hits = _collect_search_hits(observations)
        tool_section = render_tool_observation_section(observations)
        answer_guard: dict[str, Any] = {
            "generation_path": "tool_observation_only",
            "generation_attempts": 0,
        }
        evidence_section = ""
        if hits:
            candidates = build_ranked_evidence_candidates(user_text, hits)
            selection_validation = select_evidence_for_question(user_text, candidates)
            selected_ids = list(selection_validation["selected_ids"])
            selection_attempts = 0
            selection_path = str(selection_validation["selection_path"])
            synthesis_calls: list[dict[str, Any]] = []
            if selection_validation["valid"]:
                candidate_lookup = {item["evidence_id"]: item for item in candidates}
                selected_evidence = [
                    candidate_lookup[evidence_id]
                    for evidence_id in selected_ids
                    if evidence_id in candidate_lookup
                ]
                call_started = time.perf_counter()
                draft_response = model.invoke(
                    build_grounded_synthesis_messages(
                        user_text,
                        selected_evidence,
                        observations,
                    )
                )
                synthesis_calls.append(
                    {
                        "stage": "grounded_synthesis_first_pass",
                        "latency_seconds": time.perf_counter() - call_started,
                    }
                )
                draft = _response_text(draft_response)
                draft_validation = validate_grounded_draft(
                    draft,
                    selected_evidence,
                    question=user_text,
                )
                synthesis_validations = [draft_validation]
                synthesis_attempts = 1
                generation_path = "grounded_synthesis"
                if should_retry_grounded_synthesis(draft_validation):
                    call_started = time.perf_counter()
                    draft_response = model.invoke(
                        build_grounded_synthesis_retry_messages(
                            user_text,
                            selected_evidence,
                            observations,
                            draft,
                            draft_validation,
                        )
                    )
                    synthesis_calls.append(
                        {
                            "stage": "grounded_synthesis_retry",
                            "latency_seconds": time.perf_counter() - call_started,
                        }
                    )
                    draft = _response_text(draft_response)
                    draft_validation = validate_grounded_draft(
                        draft,
                        selected_evidence,
                        question=user_text,
                    )
                    synthesis_validations.append(draft_validation)
                    synthesis_attempts = 2
                    generation_path = "grounded_synthesis_retry"
                if draft_validation["valid"]:
                    evidence_section = (
                        f"{draft.strip()}\n\n"
                        "## 已知边界\n\n"
                        "以上知识解释来自本轮已校验证据，不能替代现场检查和人工复核，"
                        "也不能用于推断精确剩余寿命。"
                    )
                else:
                    evidence_section = render_structured_evidence_answer(
                        user_text,
                        candidates,
                        selected_ids,
                        selection_validation.get("slot_assignments", {}),
                    )
                    generation_path = "structured_evidence_answer"
                answer_guard = {
                    "generation_path": generation_path,
                    "selection_path": selection_path,
                    "selection_attempts": selection_attempts,
                    "selection_validation": selection_validation,
                    "synthesis_attempts": synthesis_attempts,
                    "synthesis_validation": draft_validation,
                    "synthesis_validations": synthesis_validations,
                    "llm_calls": synthesis_calls,
                    "final_citation_validation": (
                        validate_answer_citations(evidence_section, hits)
                        if generation_path == "structured_evidence_answer"
                        else {
                            "valid": draft_validation["valid"],
                            "claim_citation_coverage": draft_validation["claim_citation_coverage"],
                        }
                    ),
                }
            else:
                evidence_section = render_safe_fallback(
                    user_text,
                    candidates,
                    selection_validation,
                )
                generation_path = "safe_fallback"
                answer_guard = {
                    "generation_path": generation_path,
                    "selection_path": selection_path,
                    "selection_attempts": selection_attempts,
                    "selection_validation": selection_validation,
                    "synthesis_attempts": 0,
                    "llm_calls": [],
                    "final_citation_validation": validate_answer_citations(
                        evidence_section,
                        hits,
                    ),
                }
        elif not tool_section:
            evidence_section = (
                "## 当前证据不足\n\n"
                "本轮没有获得可用的工具观察或知识证据，请补充设备类型、"
                "现场现象或有效技术资料。"
            )
        content = "\n\n".join(section for section in (tool_section, evidence_section) if section)
        planning_metadata = state.get("planning_metadata") or {}
        observation_metadata = state.get("observation_metadata") or {}
        if planning_metadata.get("generation_path") == "deterministic_fallback":
            content = f"> 规划降级：模型两次未返回合格计划，本轮使用确定性安全路由。\n\n{content}"
        llm_calls = [
            *list(planning_metadata.get("llm_calls") or []),
            *list(observation_metadata.get("llm_calls") or []),
            *list(answer_guard.get("llm_calls") or []),
        ]
        metadata = {
            "planning": planning_metadata,
            "observation": observation_metadata,
            "answer_guard": answer_guard,
            "tool_step_count": state.get("tool_step_count", 0),
            "llm_calls": llm_calls,
            "llm_call_count": len(llm_calls),
            "llm_latency_seconds": sum(
                float(item.get("latency_seconds") or 0.0) for item in llm_calls
            ),
        }
        return {
            "messages": [
                AIMessage(
                    content=content,
                    response_metadata={"equipdoc_agentic": metadata},
                )
            ],
            "answer_metadata": metadata,
            "retrieval_hits": hits[:5],
        }

    graph = StateGraph(AgenticState)
    graph.add_node("policy_gate", policy_gate_node)
    graph.add_node("safety_response", safety_response_node)
    graph.add_node("planner", planner_node)
    graph.add_node("prepare_tool", prepare_tool_node)
    graph.add_node("review", review_node)
    graph.add_node("tools", tool_node)
    graph.add_node("diagnosis_evidence", diagnosis_evidence_node)
    graph.add_node("observer", observer_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("cancel", cancel_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_edge(START, "policy_gate")
    graph.add_conditional_edges("policy_gate", route_after_policy)
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_conditional_edges("prepare_tool", route_pending_tool)
    graph.add_conditional_edges("review", after_review)
    graph.add_conditional_edges("tools", route_after_tool)
    graph.add_edge("diagnosis_evidence", "tools")
    graph.add_conditional_edges("observer", route_after_observer)
    graph.add_edge("safety_response", END)
    graph.add_edge("clarification", END)
    graph.add_edge("cancel", END)
    graph.add_edge("synthesizer", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
