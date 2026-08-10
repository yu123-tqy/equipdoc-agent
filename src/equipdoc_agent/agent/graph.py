from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from ..config import Settings
from ..rag import KnowledgeRetriever
from ..tools import analyze_bearing_signal
from .policy import requests_diagnosis, should_run_diagnosis
from .knowledge_answer import (
    build_ranked_evidence_candidates,
    build_citation_retry_messages,
    build_full_rag_messages,
    extract_evidence_selection,
    render_extractive_fallback,
    render_selected_evidence,
    validate_answer_citations,
    validate_evidence_selection,
)
from .reporting import render_diagnosis_report
from .safety import assess_high_risk_question


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    signal_path: str
    review_result: str
    retrieval_hits: list[dict]


def _last_human_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _parse_tool_result(content) -> dict:
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(str(content))
    except (TypeError, json.JSONDecodeError):
        return {"status": "error", "error": str(content)}
    return parsed if isinstance(parsed, dict) else {"status": "error", "error": str(parsed)}


def _fault_filter(fault_type: str) -> dict[str, str] | None:
    if "外圈" in fault_type:
        return {"equipment": "bearing", "fault_type": "outer_race"}
    if "内圈" in fault_type:
        return {"equipment": "bearing", "fault_type": "inner_race"}
    if "滚动体" in fault_type:
        return {"equipment": "bearing", "fault_type": "ball"}
    return None


def _knowledge_filter(question: str) -> dict[str, str] | None:
    mappings = (
        ("外圈", {"equipment": "bearing", "fault_type": "outer_race"}),
        ("内圈", {"equipment": "bearing", "fault_type": "inner_race"}),
        ("滚动体", {"equipment": "bearing", "fault_type": "ball"}),
        ("保持架", {"equipment": "bearing", "fault_type": "cage"}),
        ("CWRU", {"equipment": "bearing", "fault_type": "dataset"}),
        ("动力电池", {"equipment": "traction_battery"}),
        ("BMS", {"equipment": "traction_battery"}),
        ("管道", {"equipment": "pipeline"}),
        ("汽蚀", {"equipment": "pump_gearbox"}),
        ("齿轮", {"equipment": "pump_gearbox"}),
    )
    matches = [
        (question.find(keyword), filters)
        for keyword, filters in mappings
        if keyword in question
    ]
    return min(matches, key=lambda item: item[0])[1] if matches else None


def _review_call_payload(call: dict) -> dict:
    """Build a reviewer-facing tool call without exposing server filesystem paths."""
    args = dict(call.get("args") or {})
    signal_path = args.pop("signal_path", None)
    if signal_path:
        normalized = str(signal_path).replace("\\", "/")
        args["signal_file"] = normalized.rsplit("/", 1)[-1]
    return {"name": call.get("name"), "args": args}


def build_graph(settings: Settings | None = None, *, checkpointer=None):
    settings = settings or Settings.from_env()
    if not settings.demo_mode and settings.agentic_mode:
        from .agentic_graph import build_agentic_graph

        if checkpointer is None:
            return build_agentic_graph(settings)
        return build_agentic_graph(settings, checkpointer=checkpointer)
    retriever_holder: dict[str, KnowledgeRetriever] = {}

    def get_retriever() -> KnowledgeRetriever | None:
        if not settings.rag_enabled or not settings.rag_chunks_path.exists():
            return None
        if "retriever" not in retriever_holder:
            retriever_holder["retriever"] = KnowledgeRetriever(settings)
        return retriever_holder["retriever"]

    @tool
    def diagnose_bearing(signal_path: str) -> dict:
        """Analyze a sandboxed .npy bearing vibration signal after human review."""
        return analyze_bearing_signal(signal_path, settings)

    tools = [diagnose_bearing]
    tools_by_name = {item.name: item for item in tools}
    llm = None
    if not settings.demo_mode:
        llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
            temperature=0,
        )

    def agent_node(state: AgentState) -> dict:
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], ToolMessage):
            result = _parse_tool_result(messages[-1].content)
            if result.get("status") == "error":
                return {"messages": [AIMessage(content=f"诊断工具执行失败：{result.get('error')}")]}
            signal_path = state.get("signal_path", "")
            evidence = []
            retriever = get_retriever()
            if retriever is not None:
                fault_type = str(result.get("fault_type", ""))
                query = f"{fault_type} 故障机理 振动特征 维修建议 风险提示"
                evidence = retriever.search(query, filters=_fault_filter(fault_type), top_k=3)
                evidence.extend(
                    retriever.search("维修决策 风险提示 不能推断剩余寿命", top_k=2)
                )
                unique = {}
                for item in evidence:
                    unique[item.get("chunk_id")] = item
                evidence = list(unique.values())[:5]
            report = render_diagnosis_report(
                result,
                signal_name=Path(signal_path).name if signal_path else "未提供",
                evidence=evidence,
            )
            return {
                "messages": [AIMessage(content=report)],
                "retrieval_hits": evidence,
            }

        user_text = _last_human_text(messages)
        signal_path = state.get("signal_path")
        safety_decision = assess_high_risk_question(user_text)
        if safety_decision is not None:
            retriever = get_retriever()
            safety_query = f"{user_text} {safety_decision.message}"
            hits = []
            if retriever and user_text:
                hits.extend(
                    retriever.search(
                        "RAG 回答边界 拒答 人工审核 证据不足 不编造",
                        filters={"fault_type": "safety"},
                        top_k=1,
                    )
                )
                hits.extend(retriever.search(safety_query, top_k=5))
                unique_hits = {}
                for item in hits:
                    unique_hits[item.get("chunk_id")] = item
                hits = list(unique_hits.values())[:5]
            snippets = "\n".join(
                f"- [{item.get('doc_id')}#{item.get('chunk_id')}]："
                f"{str(item.get('text', '')).replace(chr(10), ' ')[:180]}"
                for item in hits
            ) or "- 当前未检索到可用证据。"
            content = (
                f"## 安全边界（{safety_decision.policy_id}）\n\n"
                f"{safety_decision.message}\n\n"
                "## 检索证据\n\n"
                f"{snippets}\n\n"
                "以上为可审计的规则与原文证据，不代表已经诊断或控制真实设备。"
            )
            return {
                "messages": [AIMessage(content=content)],
                "retrieval_hits": hits,
            }

        if should_run_diagnosis(user_text, signal_path):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "diagnose_bearing",
                                "args": {"signal_path": signal_path},
                                "id": f"diagnose_{uuid4().hex}",
                            }
                        ],
                    )
                ]
            }

        if not signal_path and requests_diagnosis(user_text):
            return {
                "messages": [
                    AIMessage(
                        content="请上传需要诊断的 .npy 轴承振动信号，或勾选仓库内置演示信号。"
                    )
                ]
            }

        if settings.demo_mode:
            retriever = get_retriever()
            hits = retriever.search(user_text, top_k=5) if retriever and user_text else []
            if hits:
                snippets = "\n".join(
                    f"- [{item.get('doc_id')}#{item.get('chunk_id')}]："
                    f"{str(item.get('text', '')).replace(chr(10), ' ')[:180]}"
                    for item in hits
                )
                content = (
                    "当前为 Demo 模式，不调用7B模型。以下是知识库检索命中的项目知识片段：\n\n"
                    f"{snippets}\n\n完整生成回答需要将 EQUIPDOC_DEMO_MODE 设为 false 并配置模型服务。"
                )
            else:
                content = "当前为 Demo 模式。请上传 .npy 信号进行固定案例演示，或配置模型服务后进行知识问答。"
            return {
                "messages": [AIMessage(content=content)],
                "retrieval_hits": hits,
            }

        assert llm is not None
        retriever = get_retriever()
        hits = []
        if retriever and user_text:
            filters = _knowledge_filter(user_text)
            if filters:
                hits.extend(
                    {**item, "focused_match": True}
                    for item in retriever.search(user_text, filters=filters, top_k=3)
                )
            hits.extend(retriever.search(user_text, top_k=5))
            unique_hits = {}
            for item in hits:
                unique_hits.setdefault(item.get("chunk_id"), item)
            hits = list(unique_hits.values())[:5]
        if not hits:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "当前证据不足，知识库未检索到可用资料。请补充设备类型、"
                            "现场现象或有效技术文档后再判断。"
                        )
                    )
                ],
                "retrieval_hits": [],
            }
        candidates = build_ranked_evidence_candidates(user_text, hits)
        response = llm.invoke(build_full_rag_messages(user_text, hits))
        selected_ids = extract_evidence_selection(str(response.content))
        selection_validation = validate_evidence_selection(selected_ids, candidates)
        generation_path = "first_pass"
        attempts = 1
        if not selection_validation["valid"]:
            response = llm.invoke(
                build_citation_retry_messages(user_text, hits, str(response.content))
            )
            selected_ids = extract_evidence_selection(str(response.content))
            selection_validation = validate_evidence_selection(selected_ids, candidates)
            generation_path = "retry"
            attempts = 2

        if selection_validation["valid"]:
            content = render_selected_evidence(candidates, selected_ids)
        else:
            content = render_extractive_fallback(hits, question=user_text)
            generation_path = "extractive_fallback"
        validation = validate_answer_citations(content, hits)

        response_metadata = dict(getattr(response, "response_metadata", None) or {})
        response_metadata["equipdoc_answer_guard"] = {
            "generation_path": generation_path,
            "generation_attempts": attempts,
            "selection_validation": selection_validation,
            "final_citation_validation": validation,
        }
        message_kwargs = {"content": content, "response_metadata": response_metadata}
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            message_kwargs["usage_metadata"] = usage_metadata
        return {
            "messages": [AIMessage(**message_kwargs)],
            "retrieval_hits": hits,
        }

    def should_continue(state: AgentState):
        last_message = state.get("messages", [])[-1]
        if getattr(last_message, "tool_calls", None):
            return "review"
        return END

    def review_node(state: AgentState) -> dict:
        calls = state.get("messages", [])[-1].tool_calls
        decision = interrupt(
            {
                "type": "tool_review",
                "requested_tools": [_review_call_payload(item) for item in calls],
                "notice": "Approve runs a read-only diagnostic tool; Reject cancels the call.",
            }
        )
        return {"review_result": decision}

    def after_review(state: AgentState):
        return "tools" if state.get("review_result") == "approve" else "cancel"

    def tool_node(state: AgentState) -> dict:
        outputs = []
        for call in state.get("messages", [])[-1].tool_calls:
            name = call.get("name")
            if name not in tools_by_name:
                payload = {"status": "error", "error": f"Unknown tool: {name}"}
            else:
                try:
                    payload = tools_by_name[name].invoke(call.get("args") or {})
                except Exception as exc:
                    payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            outputs.append(
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=call.get("id", f"tool_{uuid4().hex}"),
                )
            )
        return {"messages": outputs}

    def cancel_node(_: AgentState) -> dict:
        return {"messages": [AIMessage(content="已根据人工审核取消本次诊断工具调用。")]} 

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("review", review_node)
    graph.add_node("tools", tool_node)
    graph.add_node("cancel", cancel_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_conditional_edges("review", after_review)
    graph.add_edge("tools", "agent")
    graph.add_edge("cancel", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
