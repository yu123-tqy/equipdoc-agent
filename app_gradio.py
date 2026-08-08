from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
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
from equipdoc_agent.health import collect_health
from equipdoc_agent.networking import find_available_port
from equipdoc_agent.privacy import public_exception_message
from equipdoc_agent.retrieval_display import (
    render_retrieval_hits_markdown,
    sanitize_retrieval_hits,
)
from equipdoc_agent.runtime_cleanup import cleanup_stale_uploads


SETTINGS = Settings.from_env(ROOT)
SETTINGS.ensure_runtime_dirs()
cleanup_stale_uploads(SETTINGS)
AGENT = build_graph(SETTINGS)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _last_content(result: dict) -> str:
    messages = result.get("messages", [])
    return str(getattr(messages[-1], "content", "")) if messages else ""


def _interrupt_payload(result: dict):
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    return getattr(first, "value", first)


def _retrieval_view(result: dict):
    hits = sanitize_retrieval_hits(result.get("retrieval_hits") or [])
    return (
        hits,
        gr.update(visible=bool(hits), interactive=bool(hits)),
        gr.update(value="", visible=False),
    )


def show_retrieval_hits(hits):
    safe_hits = sanitize_retrieval_hits(hits)
    return gr.update(
        value=render_retrieval_hits_markdown(safe_hits),
        visible=bool(safe_hits),
    )


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


def begin_submission():
    """Put the UI in a cancellable busy state before an agent request starts."""
    return (
        "处理中，可点击“停止当前任务”取消本次等待。",
        "",
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        [],
        gr.update(visible=False, interactive=False),
        gr.update(value="", visible=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def stop_current_task():
    """Reset visible task state; Gradio cancels the linked running event."""
    return (
        "",
        "已停止当前任务。请修改问题后重新提交。",
        "",
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        [],
        gr.update(visible=False, interactive=False),
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def submit(question, uploaded_file, use_sample, thread_id):
    thread_id = thread_id or str(uuid4())
    try:
        signal_path = _stage_signal(uploaded_file, bool(use_sample))
        text = (question or "").strip()
        if not text and signal_path:
            text = "请诊断这段轴承振动信号并给出有依据的处理建议。"
        if not text:
            raise ValueError("请输入问题，或者选择/上传一个信号文件。")
        payload = {
            "messages": [HumanMessage(content=text)],
            # Explicitly clear a signal retained by the same LangGraph thread
            # when the user unchecks/removes the file on a later turn.
            "signal_path": str(signal_path) if signal_path else "",
            # Clear the previous turn's trace before the graph records this turn.
            "retrieval_hits": [],
        }
        result = AGENT.invoke(payload, config=_config(thread_id))
        payload = _interrupt_payload(result)
        if payload:
            return (
                thread_id,
                "等待人工审核",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "",
                gr.update(interactive=True),
                gr.update(interactive=True),
                [],
                gr.update(visible=False, interactive=False),
                gr.update(value="", visible=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
            )
        retrieval_state, retrieval_button, retrieval_box = _retrieval_view(result)
        return (
            thread_id,
            "已完成",
            "",
            _last_content(result),
            gr.update(interactive=False),
            gr.update(interactive=False),
            retrieval_state,
            retrieval_button,
            retrieval_box,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    except Exception as exc:
        return (
            thread_id,
            f"请求失败：{type(exc).__name__}",
            "",
            public_exception_message(exc, project_root=SETTINGS.project_root),
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            gr.update(visible=False, interactive=False),
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )


def resume_review(decision: str, thread_id: str):
    if not thread_id:
        return (
            "没有待审核任务",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            gr.update(visible=False, interactive=False),
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    try:
        result = AGENT.invoke(Command(resume=decision), config=_config(thread_id))
        retrieval_state, retrieval_button, retrieval_box = _retrieval_view(result)
        return (
            "已完成",
            "",
            _last_content(result),
            gr.update(interactive=False),
            gr.update(interactive=False),
            retrieval_state,
            retrieval_button,
            retrieval_box,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    except Exception as exc:
        return (
            f"执行失败：{type(exc).__name__}",
            "",
            public_exception_message(exc, project_root=SETTINGS.project_root),
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            gr.update(visible=False, interactive=False),
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
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

with gr.Blocks(title="EquipDoc-Agent") as demo:
    gr.Markdown("# EquipDoc-Agent｜机电设备智能运维 Agent")
    gr.Markdown(MODE_NOTICE)
    thread_state = gr.State("")
    retrieval_state = gr.State([])

    with gr.Row():
        question_box = gr.Textbox(
            label="运维问题",
            lines=4,
            value="请诊断这段轴承振动信号，并给出判断依据和处理建议。",
        )
        with gr.Column():
            use_sample = gr.Checkbox(label="使用仓库内置演示信号", value=True)
            upload = gr.File(label="或上传 .npy 信号", file_types=[".npy"], type="filepath")

    with gr.Row():
        submit_button = gr.Button("提交", variant="primary")
        stop_button = gr.Button("停止当前任务", variant="stop", interactive=False)
        approve_button = gr.Button("Approve", interactive=False)
        reject_button = gr.Button("Reject", interactive=False)

    status_box = gr.Textbox(label="状态", interactive=False)
    review_box = gr.Code(label="待审核工具调用", language="json")
    report_box = gr.Markdown()
    retrieval_button = gr.Button(
        "查看本次召回 Top 5",
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
        report_box,
        approve_button,
        reject_button,
        retrieval_state,
        retrieval_button,
        retrieval_box,
        submit_button,
        stop_button,
    ]
    request_outputs = [
        thread_state,
        status_box,
        review_box,
        report_box,
        approve_button,
        reject_button,
        retrieval_state,
        retrieval_button,
        retrieval_box,
        submit_button,
        stop_button,
    ]
    review_outputs = [
        status_box,
        review_box,
        report_box,
        approve_button,
        reject_button,
        retrieval_state,
        retrieval_button,
        retrieval_box,
        submit_button,
        stop_button,
    ]

    submit_start_event = submit_button.click(
        begin_submission,
        outputs=request_start_outputs,
        queue=False,
    )
    submit_event = submit_start_event.then(
        submit,
        inputs=[question_box, upload, use_sample, thread_state],
        outputs=request_outputs,
        concurrency_limit=2,
        concurrency_id="agent_requests",
    )
    approve_start_event = approve_button.click(
        begin_submission,
        outputs=request_start_outputs,
        queue=False,
    )
    approve_event = approve_start_event.then(
        lambda thread_id: resume_review("approve", thread_id),
        inputs=[thread_state],
        outputs=review_outputs,
        concurrency_limit=2,
        concurrency_id="agent_requests",
    )
    reject_start_event = reject_button.click(
        begin_submission,
        outputs=request_start_outputs,
        queue=False,
    )
    reject_event = reject_start_event.then(
        lambda thread_id: resume_review("reject", thread_id),
        inputs=[thread_state],
        outputs=review_outputs,
        concurrency_limit=2,
        concurrency_id="agent_requests",
    )
    stop_button.click(
        stop_current_task,
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
    )
