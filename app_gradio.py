from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from threading import Lock
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from equipdoc_agent.agent import build_graph
from equipdoc_agent.config import Settings
from equipdoc_agent.conversation_store import ConversationStore
from equipdoc_agent.health import collect_health
from equipdoc_agent.networking import find_available_port
from equipdoc_agent.persistence import create_checkpointer, delete_checkpoint_threads
from equipdoc_agent.privacy import public_exception_message
from equipdoc_agent.retrieval_display import (
    render_retrieval_hits_markdown,
    sanitize_retrieval_hits,
)
from equipdoc_agent.runtime_cleanup import cleanup_stale_uploads


SETTINGS = Settings.from_env(ROOT)
SETTINGS.ensure_runtime_dirs()
cleanup_stale_uploads(SETTINGS)
STORE = ConversationStore(
    SETTINGS.conversation_database,
    recent_context_messages=SETTINGS.recent_context_messages,
    summary_max_chars=SETTINGS.conversation_summary_max_chars,
    compaction_turns=SETTINGS.memory_compaction_turns,
)
STORE.recover_interrupted_runs(all_owners=True)
CHECKPOINTER = create_checkpointer(SETTINGS.checkpoint_database)
AGENT = build_graph(SETTINGS, checkpointer=CHECKPOINTER)
_RETENTION_LOCK = Lock()
_RETENTION_OWNERS: set[str] = set()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _owner_id(request: gr.Request = None) -> str:
    username = str(getattr(request, "username", "") or "").strip()
    if SETTINGS.auth_enabled and not username:
        raise PermissionError("登录状态无效，请重新登录。")
    return username or "local"


def _owner_store(request: gr.Request = None) -> ConversationStore:
    owner_id = _owner_id(request)
    store = STORE if STORE.owner_id == owner_id else STORE.for_owner(owner_id)
    with _RETENTION_LOCK:
        if owner_id not in _RETENTION_OWNERS:
            purged = store.purge_archived(SETTINGS.conversation_retention_days)
            delete_checkpoint_threads(CHECKPOINTER, purged["thread_ids"])
            _RETENTION_OWNERS.add(owner_id)
    return store


def _last_content(result: dict) -> str:
    messages = result.get("messages", [])
    return str(getattr(messages[-1], "content", "")) if messages else ""


def _interrupt_payload(result: dict):
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    return getattr(first, "value", first)


def _uploaded_path(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None
    raw = getattr(uploaded_file, "name", uploaded_file)
    return Path(str(raw)).expanduser().resolve(strict=True)


def _stage_signal(uploaded_file, use_sample: bool) -> Path | None:
    cleanup_stale_uploads(SETTINGS)
    if use_sample:
        sample = SETTINGS.sample_root / "test_signal.npy"
        if not sample.exists():
            raise FileNotFoundError("Bundled sample is missing. Run equipdoc-health.")
        return sample.resolve()
    source = _uploaded_path(uploaded_file)
    if source is None:
        return None
    if source.suffix.lower() != ".npy":
        raise ValueError("Only .npy files are accepted.")
    if source.stat().st_size > SETTINGS.max_upload_bytes:
        raise ValueError("Uploaded file exceeds the configured size limit.")
    destination = SETTINGS.upload_root / f"{uuid4().hex}.npy"
    shutil.copy2(source, destination)
    return destination.resolve()


def _history_page(
    store: ConversationStore,
    *,
    search: str,
    include_archived: bool,
    page: int,
    current_id: str = "",
) -> tuple[list[tuple[str, str]], int, int, int]:
    page_size = SETTINGS.history_page_size
    total = store.count_conversations(search=search, include_archived=include_archived)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(int(page or 0), pages - 1))
    items = store.list_conversations(
        search=search,
        include_archived=include_archived,
        limit=page_size,
        offset=page * page_size,
    )
    rows: list[tuple[str, dict]] = []
    label_counts: dict[str, int] = {}
    for item in items:
        updated = str(item["updated_at"])
        stamp = updated[5:19].replace("T", " ") if len(updated) >= 19 else updated
        markers = []
        if item.get("status") == "awaiting_review":
            markers.append("待审核")
        if item.get("archived_at"):
            markers.append("已归档")
        marker = f" · {'/'.join(markers)}" if markers else ""
        label = f"{item['title']} · {stamp}{marker}"
        rows.append((label, item))
        label_counts[label] = label_counts.get(label, 0) + 1
    choices: list[tuple[str, str]] = []
    for label, item in rows:
        if label_counts[label] > 1:
            label = f"{label} · {str(item['id'])[:6]}"
        choices.append((label, str(item["id"])))
    return choices, total, page, pages


def _chat_messages(store: ConversationStore, conversation_id: str) -> list[dict[str, str]]:
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in store.list_messages(conversation_id)
    ]


def _latest_retrieval_hits(
    store: ConversationStore, conversation_id: str
) -> list[dict]:
    for item in reversed(store.list_messages(conversation_id)):
        if item.get("role") != "assistant":
            continue
        hits = (item.get("metadata") or {}).get("retrieval_hits")
        return sanitize_retrieval_hits(hits or [])
    return []


def _ensure_conversation(
    store: ConversationStore,
    conversation_id: str | None,
    *,
    search: str = "",
    include_archived: bool = False,
) -> str:
    if conversation_id and store.get_conversation(conversation_id):
        return conversation_id
    conversations = store.list_conversations(
        search=search,
        include_archived=include_archived,
        limit=1,
    )
    if conversations:
        return str(conversations[0]["id"])
    return store.create_conversation()


def _conversation_view(
    conversation_id: str | None,
    *,
    store: ConversationStore | None = None,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    status: str | None = None,
):
    store = store or STORE
    conversation_id = _ensure_conversation(
        store,
        conversation_id,
        search=search,
        include_archived=include_archived,
    )
    conversation = store.get_conversation(conversation_id) or {"title": "新对话"}
    active = store.get_active_run(conversation_id)
    review_payload = (
        active.get("pending_review")
        if active and active.get("status") == "awaiting_review"
        else None
    )
    is_archived = bool(conversation.get("archived_at"))
    if status is None:
        if review_payload:
            status = "等待人工审核"
        elif active:
            status = "任务正在运行"
        elif is_archived:
            status = "该会话已归档，可恢复或永久删除"
        else:
            status = ""
    hits = _latest_retrieval_hits(store, conversation_id)
    is_busy = bool(active)
    awaiting_review = bool(review_payload)
    choices, total, page, pages = _history_page(
        store,
        search=search,
        include_archived=include_archived,
        page=page,
        current_id=conversation_id,
    )
    can_edit = not is_busy and not is_archived
    return (
        conversation_id,
        gr.update(choices=choices, value=conversation_id, interactive=not is_busy),
        str(conversation["title"]),
        _chat_messages(store, conversation_id),
        status,
        json.dumps(review_payload, ensure_ascii=False, indent=2) if review_payload else "",
        gr.update(interactive=awaiting_review),
        gr.update(interactive=awaiting_review),
        hits,
        gr.update(visible=bool(hits), interactive=bool(hits)),
        gr.update(value="", visible=False),
        gr.update(interactive=can_edit),
        gr.update(interactive=is_busy),
        gr.update(interactive=not is_busy),
        gr.update(interactive=can_edit),
        gr.update(interactive=can_edit),
        gr.update(interactive=not is_busy and is_archived),
        gr.update(interactive=not is_busy),
        gr.update(value=False, interactive=not is_busy),
        page,
        f"第 {page + 1}/{pages} 页 · 共 {total} 个会话",
        gr.update(interactive=not is_busy and page > 0),
        gr.update(interactive=not is_busy and page + 1 < pages),
        search,
    )


def initialize_history(
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    owner_notice = f"已登录：{store.owner_id}" if SETTINGS.auth_enabled else "本地单用户模式"
    return _conversation_view(
        None,
        store=store,
        search=search,
        include_archived=include_archived,
        page=page,
        status=owner_notice,
    )


def load_conversation(
    conversation_id: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    return _conversation_view(
        conversation_id,
        store=_owner_store(request),
        search=search,
        include_archived=include_archived,
        page=page,
    )


def filter_history(
    search: str,
    include_archived: bool,
    page: int,
    conversation_id: str,
    request: gr.Request = None,
):
    store = _owner_store(request)
    candidates = store.list_conversations(
        search=search,
        include_archived=include_archived,
        limit=1,
    )
    selected_id = str(candidates[0]["id"]) if candidates else conversation_id
    return _conversation_view(
        selected_id,
        store=store,
        search=search,
        include_archived=include_archived,
        page=0,
    )


def change_history_page(
    delta: int,
    search: str,
    include_archived: bool,
    page: int,
    conversation_id: str,
    request: gr.Request = None,
):
    store = _owner_store(request)
    target_page = max(0, int(page or 0) + int(delta))
    candidates = store.list_conversations(
        search=search,
        include_archived=include_archived,
        limit=1,
        offset=target_page * SETTINGS.history_page_size,
    )
    selected_id = str(candidates[0]["id"]) if candidates else conversation_id
    return _conversation_view(
        selected_id,
        store=store,
        search=search,
        include_archived=include_archived,
        page=target_page,
    )


def previous_history_page(
    search: str,
    include_archived: bool,
    page: int,
    conversation_id: str,
    request: gr.Request = None,
):
    return change_history_page(
        -1, search, include_archived, page, conversation_id, request
    )


def next_history_page(
    search: str,
    include_archived: bool,
    page: int,
    conversation_id: str,
    request: gr.Request = None,
):
    return change_history_page(
        1, search, include_archived, page, conversation_id, request
    )


def new_conversation(
    search: str = "",
    include_archived: bool = False,
    request: gr.Request = None,
):
    store = _owner_store(request)
    return _conversation_view(
        store.create_conversation(),
        store=store,
        search="",
        include_archived=include_archived,
        page=0,
        status="已新建对话",
    )


def rename_conversation(
    conversation_id: str,
    title: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    conversation_id = _ensure_conversation(store, conversation_id)
    store.rename_conversation(conversation_id, title)
    return _conversation_view(
        conversation_id,
        store=store,
        search=search,
        include_archived=include_archived,
        page=page,
        status="会话已重命名",
    )


def archive_conversation(
    conversation_id: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    conversation_id = _ensure_conversation(store, conversation_id)
    if store.get_active_run(conversation_id):
        return _conversation_view(
            conversation_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status="请先完成或停止当前任务再归档。",
        )
    store.archive_conversation(conversation_id)
    if include_archived:
        next_id = conversation_id
    else:
        remaining = store.list_conversations(search=search, limit=1)
        if remaining:
            next_id = str(remaining[0]["id"])
        else:
            next_id = store.create_conversation()
            search = ""
    return _conversation_view(
        next_id,
        store=store,
        search=search,
        include_archived=include_archived,
        page=page,
        status="会话已归档",
    )


def restore_archived_conversation(
    conversation_id: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    conversation_id = _ensure_conversation(store, conversation_id, include_archived=True)
    restored = store.restore_conversation(conversation_id)
    return _conversation_view(
        conversation_id,
        store=store,
        search=search,
        include_archived=include_archived,
        page=page,
        status="会话已恢复" if restored else "当前会话未归档",
    )


def permanently_delete_conversation(
    conversation_id: str,
    confirmed: bool,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    conversation_id = _ensure_conversation(store, conversation_id, include_archived=True)
    if not confirmed:
        return _conversation_view(
            conversation_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status="请先勾选“确认永久删除”；删除后无法恢复。",
        )
    try:
        thread_ids = store.delete_conversation(conversation_id)
    except RuntimeError as exc:
        return _conversation_view(
            conversation_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status=str(exc),
        )
    delete_checkpoint_threads(CHECKPOINTER, thread_ids)
    remaining = store.list_conversations(
        search=search,
        include_archived=include_archived,
        limit=1,
    )
    if remaining:
        next_id = str(remaining[0]["id"])
    else:
        next_id = store.create_conversation()
        search = ""
    return _conversation_view(
        next_id,
        store=store,
        search=search,
        include_archived=include_archived,
        page=0,
        status="会话及其 Agent 检查点已永久删除",
    )


def restore_conversation_after_clear(
    conversation_id: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    return _conversation_view(
        conversation_id,
        store=_owner_store(request),
        search=search,
        include_archived=include_archived,
        page=page,
        status="历史记录已持久化；如需移除整段会话，请使用归档或永久删除。",
    )


def show_retrieval_hits(hits):
    safe_hits = sanitize_retrieval_hits(hits)
    return gr.update(
        value=render_retrieval_hits_markdown(safe_hits),
        visible=bool(safe_hits),
    )


def begin_submission():
    """Put the UI in a cancellable busy state before an agent request starts."""
    return (
        "处理中，可点击“停止当前任务”取消本次等待。",
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        [],
        gr.update(visible=False, interactive=False),
        gr.update(value="", visible=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=False, interactive=False),
    )


def _agent_payload(store: ConversationStore, conversation_id: str, text_value: str, signal_path):
    memory = store.latest_agent_memory(conversation_id)
    memory["conversation_context"] = store.conversation_context(conversation_id)
    return {
        "messages": [HumanMessage(content=text_value)],
        "signal_path": str(signal_path) if signal_path else "",
        "retrieval_hits": [],
        "session_memory": memory,
    }


def _result_metadata(result: dict, hits: list[dict], **extra) -> dict:
    metadata = {"retrieval_hits": hits, **extra}
    memory = result.get("session_memory")
    if isinstance(memory, dict):
        metadata["agent_memory"] = memory
    return metadata


def submit(
    question,
    uploaded_file,
    use_sample,
    thread_id,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    thread_id = _ensure_conversation(store, thread_id)
    run_id = ""
    try:
        signal_path = _stage_signal(uploaded_file, bool(use_sample))
        text_value = (question or "").strip()
        if not text_value and signal_path:
            text_value = "请诊断这段轴承振动信号并给出有依据的处理建议。"
        if not text_value:
            raise ValueError("请输入问题，或者选择/上传一个信号文件。")
        run = store.start_run(
            thread_id,
            user_content=text_value,
            message_metadata={"signal_file": Path(signal_path).name if signal_path else ""},
        )
        run_id = run["run_id"]
        result = AGENT.invoke(
            _agent_payload(store, thread_id, text_value, signal_path),
            config=_config(store.get_agent_thread_id(thread_id)),
        )
        interrupt_payload = _interrupt_payload(result)
        if interrupt_payload:
            if store.mark_awaiting_review(run_id, interrupt_payload):
                return _conversation_view(
                    thread_id,
                    store=store,
                    search=search,
                    include_archived=include_archived,
                    page=page,
                    status="等待人工审核",
                )
            return _conversation_view(
                thread_id,
                store=store,
                search=search,
                include_archived=include_archived,
                page=page,
                status="任务已停止，审核请求未写入历史。",
            )
        hits = sanitize_retrieval_hits(result.get("retrieval_hits") or [])
        completed = store.complete_run(
            run_id,
            assistant_content=_last_content(result),
            metadata=_result_metadata(result, hits),
        )
        return _conversation_view(
            thread_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status="已完成" if completed else "已停止当前任务，后台结果已丢弃。",
        )
    except Exception as exc:
        public_message = public_exception_message(exc, project_root=SETTINGS.project_root)
        if run_id:
            store.fail_run(run_id, public_message)
        return _conversation_view(
            thread_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status=f"请求失败：{type(exc).__name__}",
        )


def resume_review(
    decision: str,
    thread_id: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    thread_id = _ensure_conversation(store, thread_id)
    active = store.get_active_run(thread_id)
    if not active or active.get("status") != "awaiting_review":
        return _conversation_view(
            thread_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status="没有待审核任务",
        )


    run_id = str(active["id"])
    if not store.begin_review_resume(run_id):
        return _conversation_view(
            thread_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status="待审核任务状态已变化，请刷新后重试。",
        )
    try:
        result = AGENT.invoke(
            Command(resume=decision),
            config=_config(store.get_agent_thread_id(thread_id)),
        )
        interrupt_payload = _interrupt_payload(result)
        if interrupt_payload:
            store.mark_awaiting_review(run_id, interrupt_payload)
            return _conversation_view(
                thread_id,
                store=store,
                search=search,
                include_archived=include_archived,
                page=page,
                status="等待人工审核",
            )
        hits = sanitize_retrieval_hits(result.get("retrieval_hits") or [])
        completed = store.complete_run(
            run_id,
            assistant_content=_last_content(result),
            metadata=_result_metadata(result, hits, review_decision=decision),
        )
        return _conversation_view(
            thread_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status="已完成" if completed else "已停止当前任务，后台结果已丢弃。",
        )
    except Exception as exc:
        public_message = public_exception_message(exc, project_root=SETTINGS.project_root)
        store.fail_run(run_id, public_message)
        return _conversation_view(
            thread_id,
            store=store,
            search=search,
            include_archived=include_archived,
            page=page,
            status=f"执行失败：{type(exc).__name__}",
        )


def approve_review(
    thread_id: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    return resume_review(
        "approve", thread_id, search, include_archived, page, request
    )


def reject_review(
    thread_id: str,
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    return resume_review(
        "reject", thread_id, search, include_archived, page, request
    )


def stop_current_task(
    thread_id: str = "",
    search: str = "",
    include_archived: bool = False,
    page: int = 0,
    request: gr.Request = None,
):
    store = _owner_store(request)
    thread_id = _ensure_conversation(store, thread_id)
    stopped = store.cancel_active_run(thread_id)
    message = (
        "已停止当前任务。请修改问题后重新提交。"
        if stopped
        else "当前没有运行中的任务。"
    )
    return _conversation_view(
        thread_id,
        store=store,
        search=search,
        include_archived=include_archived,
        page=page,
        status=message,
    )


HEALTH = collect_health(SETTINGS, public=True)
MODE_NOTICE = (
    "⚠️ 当前为 **Demo 模式**：无需7B模型；故障类别是固定案例回放，不能用于真实诊断。"
    if SETTINGS.demo_mode
    else (
        "当前为 **Full Agentic 模式**：模型执行受约束规划，诊断工具仍需人工审核。"
        if SETTINGS.agentic_mode
        else "当前为 **Full P2 基线模式**：将调用配置的模型与轴承 CNN。"
    )
)
ACCOUNT_NOTICE = (
    "🔐 已启用登录；历史会话按账号隔离。"
    if SETTINGS.auth_enabled
    else "🔒 本地单用户模式；设置 `EQUIPDOC_AUTH_USERS` 可启用账号隔离。"
)


with gr.Blocks(title="EquipDoc-Agent") as demo:
    gr.Markdown("# EquipDoc-Agent｜机电设备智能运维 Agent")
    gr.Markdown(MODE_NOTICE)
    gr.Markdown(ACCOUNT_NOTICE)
    thread_state = gr.State("")
    retrieval_state = gr.State([])
    page_state = gr.State(0)

    with gr.Row():
        with gr.Column(scale=1, min_width=290):
            new_button = gr.Button("＋ 新建对话", variant="primary")
            search_box = gr.Textbox(
                label="搜索历史",
                placeholder="搜索标题或对话内容",
                lines=1,
            )
            with gr.Row():
                search_button = gr.Button("搜索", size="sm")
                include_archived = gr.Checkbox(label="包含归档", value=False)
            conversation_list = gr.Radio(
                label="历史对话",
                choices=[],
                interactive=True,
            )
            page_label = gr.Markdown("第 1/1 页 · 共 0 个会话")
            with gr.Row():
                prev_button = gr.Button("上一页", size="sm", interactive=False)
                next_button = gr.Button("下一页", size="sm", interactive=False)
            title_box = gr.Textbox(label="当前会话标题", lines=1)
            with gr.Row():
                rename_button = gr.Button("重命名", size="sm")
                archive_button = gr.Button("归档", size="sm")
                restore_button = gr.Button("恢复", size="sm", interactive=False)
            delete_confirm = gr.Checkbox(label="确认永久删除（不可恢复）", value=False)
            delete_button = gr.Button("永久删除", variant="stop")

        with gr.Column(scale=4, min_width=520):
            chatbot = gr.Chatbot(
                label="对话内容",
                height=520,
                layout="bubble",
                placeholder="新建或选择一个对话后开始提问。",
            )
            question_box = gr.Textbox(
                label="运维问题",
                lines=3,
                value="请诊断这段轴承振动信号，并给出判断依据和处理建议。",
            )
            with gr.Row():
                use_sample = gr.Checkbox(label="使用仓库内置演示信号", value=True)
                upload = gr.File(
                    label="或上传 .npy 信号", file_types=[".npy"], type="filepath"
                )
            with gr.Row():
                submit_button = gr.Button("提交", variant="primary")
                stop_button = gr.Button("停止当前任务", variant="stop", interactive=False)
                approve_button = gr.Button("Approve", interactive=False)
                reject_button = gr.Button("Reject", interactive=False)

            status_box = gr.Textbox(label="状态", interactive=False)
            review_box = gr.Code(label="待审核工具调用", language="json")
            retrieval_button = gr.Button(
                "查看最近一次回答的召回 Top 5",
                variant="secondary",
                visible=False,
                interactive=False,
            )
            retrieval_box = gr.Markdown(visible=False)
            with gr.Accordion("启动健康检查", open=False):
                gr.Code(value=json.dumps(HEALTH, ensure_ascii=False, indent=2), language="json")

    request_start_outputs = [
        status_box,
        review_box,
        approve_button,
        reject_button,
        retrieval_state,
        retrieval_button,
        retrieval_box,
        submit_button,
        stop_button,
        conversation_list,
        new_button,
        rename_button,
        archive_button,
        restore_button,
        delete_button,
        delete_confirm,
    ]
    request_outputs = [
        thread_state,
        conversation_list,
        title_box,
        chatbot,
        status_box,
        review_box,
        approve_button,
        reject_button,
        retrieval_state,
        retrieval_button,
        retrieval_box,
        submit_button,
        stop_button,
        new_button,
        rename_button,
        archive_button,
        restore_button,
        delete_button,
        delete_confirm,
        page_state,
        page_label,
        prev_button,
        next_button,
        search_box,
    ]
    history_inputs = [search_box, include_archived, page_state]
    selection_inputs = [conversation_list, *history_inputs]
    current_inputs = [thread_state, *history_inputs]

    demo.load(
        initialize_history,
        inputs=history_inputs,
        outputs=request_outputs,
        queue=False,
    )
    conversation_list.input(
        load_conversation,
        inputs=selection_inputs,
        outputs=request_outputs,
        queue=False,
    )
    for search_trigger in (search_button.click, search_box.submit):
        search_trigger(
            filter_history,
            inputs=[search_box, include_archived, page_state, thread_state],
            outputs=request_outputs,
            queue=False,
        )
    include_archived.change(
        filter_history,
        inputs=[search_box, include_archived, page_state, thread_state],
        outputs=request_outputs,
        queue=False,
    )
    prev_button.click(
        previous_history_page,
        inputs=[search_box, include_archived, page_state, thread_state],
        outputs=request_outputs,
        queue=False,
    )
    next_button.click(
        next_history_page,
        inputs=[search_box, include_archived, page_state, thread_state],
        outputs=request_outputs,
        queue=False,
    )
    new_button.click(
        new_conversation,
        inputs=[search_box, include_archived],
        outputs=request_outputs,
        queue=False,
    )
    rename_button.click(
        rename_conversation,
        inputs=[thread_state, title_box, *history_inputs],
        outputs=request_outputs,
        queue=False,
    )
    archive_button.click(
        archive_conversation,
        inputs=current_inputs,
        outputs=request_outputs,
        queue=False,
    )
    restore_button.click(
        restore_archived_conversation,
        inputs=current_inputs,
        outputs=request_outputs,
        queue=False,
    )
    delete_button.click(
        permanently_delete_conversation,
        inputs=[thread_state, delete_confirm, *history_inputs],
        outputs=request_outputs,
        queue=False,
    )
    chatbot.clear(
        restore_conversation_after_clear,
        inputs=current_inputs,
        outputs=request_outputs,
        queue=False,
    )

    submit_start_event = submit_button.click(
        begin_submission,
        outputs=request_start_outputs,
        queue=False,
    )
    submit_event = submit_start_event.then(
        submit,
        inputs=[question_box, upload, use_sample, thread_state, *history_inputs],
        outputs=request_outputs,
        concurrency_limit=8,
        concurrency_id="agent_requests",
    )
    approve_start_event = approve_button.click(
        begin_submission,
        outputs=request_start_outputs,
        queue=False,
    )
    approve_event = approve_start_event.then(
        approve_review,
        inputs=[thread_state, *history_inputs],
        outputs=request_outputs,
        concurrency_limit=8,
        concurrency_id="agent_requests",
    )
    reject_start_event = reject_button.click(
        begin_submission,
        outputs=request_start_outputs,
        queue=False,
    )
    reject_event = reject_start_event.then(
        reject_review,
        inputs=[thread_state, *history_inputs],
        outputs=request_outputs,
        concurrency_limit=8,
        concurrency_id="agent_requests",
    )
    stop_button.click(
        stop_current_task,
        inputs=current_inputs,
        outputs=request_outputs,
        cancels=[submit_event, approve_event, reject_event],
        queue=False,
    )
    retrieval_button.click(
        show_retrieval_hits,
        inputs=[retrieval_state],
        outputs=[retrieval_box],
    )


if __name__ == "__main__":
    selected_port = find_available_port(SETTINGS.server_host, SETTINGS.server_port)
    if selected_port != SETTINGS.server_port:
        print(
            f"Port {SETTINGS.server_port} is occupied; "
            f"starting EquipDoc Agent on port {selected_port} instead."
        )
    demo.launch(
        server_name=SETTINGS.server_host,
        server_port=selected_port,
        share=SETTINGS.gradio_share,
        auth=list(SETTINGS.auth_users) or None,
        auth_message="EquipDoc-Agent：请输入配置的演示账号。",
    )
