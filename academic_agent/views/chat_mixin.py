"""对话、文件上传和 Agent 输出渲染。"""

from __future__ import annotations

import ast
import base64
import html
import json
import os
import re
from pathlib import Path

from PyQt6.QtCore import QEvent, QTimer, Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from academic_agent.infrastructure.workspace_manager import workspace_manager
from academic_agent.infrastructure.runtime_paths import output_root


class ChatMixin:
    """负责会话状态、上传文件、流式响应和对话区渲染。"""

    _TEXT_MINING_TRIGGER = re.compile(
        r"文本挖掘|文本聚类|聚类|情感分析|情感识别|关键词提取|主题词|文本预处理|分词|意图识别|响应识别|行为识别"
    )
    _IMAGE_REQUEST_TRIGGER = re.compile(
        r"画图|绘图|作图|图表|可视化|折线图|柱状图|条形图|饼图|散点图|热力图|分布图|词云|"
        r"生成图片|生成图像|显示图片|查看图片|图片结果|图片展示|plot|chart|visuali[sz]e|"
        r"\.(?:png|jpg|jpeg|webp|bmp|gif)\b",
        re.IGNORECASE,
    )

    def _ensure_text_mining_model(self) -> bool:
        """确认文本挖掘算法所需的本地模型已配置。"""
        from academic_agent.infrastructure.model_paths import model_components

        status = model_components(os.getenv("PRETRAINED_MODELS_DIR"))
        if status["has_bge"]:
            return True
        return bool(self.configure_text_mining_model())

    def eventFilter(self, obj, event):
        if obj is self.composer and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    def new_chat(self) -> None:
        if not self.ensure_authenticated():
            return
        if hasattr(self, "controller"):
            self.controller.sessions.new_session()
        else:
            self.messages.clear()
        self.chat_view.clear()
        if hasattr(self, "_add_current_session_entry"):
            self._add_current_session_entry()
        self.statusBar().showMessage("已新建对话")

    def upload_file(self) -> None:
        if not self.ensure_authenticated():
            return
        from academic_agent.infrastructure.data_profiler import profile_file
        from academic_agent.agent.session import session_store
        from academic_agent.tools.text_mining_tools import text_mining_tools

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据或文档文件",
            "",
            "Supported Files (*.csv *.tsv *.xlsx *.xls *.xlsm *.ods *.txt *.log *.md *.json *.jsonl *.parquet *.feather *.html *.htm *.pdf *.docx *.pptx *.png *.jpg *.jpeg *.webp *.bmp *.gif *.mp3 *.wav *.mp4 *.mov *.avi);;All Files (*)",
        )
        if not path:
            return
        try:
            result = text_mining_tools.load_data(file_path=path, source_scope="upload")
            if not result.get("success"):
                raise RuntimeError(result.get("error", "文件加载失败"))
            profiled = profile_file(path)
            session_store.get(self.session_id).attach_file(
                path, profiled if profiled.get("success") else result
            )
            rows = result.get("rows", "?")
            cols = result.get("columns", "?")
            self.messages.append({
                "role": "assistant",
                "content": (
                    f"已加载用户文件：{os.path.basename(path)}"
                    f"（{rows} 行 × {cols} 列）。该文件已绑定到当前会话，"
                    "即使不在项目工作区中也可以继续分析。"
                ),
                "_execution_steps": [
                    {"label": "读取用户选择的文件", "status": "completed"},
                    {"label": "识别数据规模与字段", "status": "completed"},
                    {"label": "绑定到当前会话上下文", "status": "completed"},
                ],
                "_image_paths": [],
                "_show_images": False,
                "_artifact_paths": [],
            })
            self._render_messages()
            self.statusBar().showMessage("文件已加载到当前 Agent 会话")
        except Exception as exc:
            QMessageBox.critical(self, "上传失败", str(exc))

    def show_image_file(self, image_path: str | None = None) -> None:
        """选择工作区图片，并复用对话区现有图片渲染逻辑展示。"""
        if not self.ensure_authenticated():
            return
        if image_path is None:
            image_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择要显示的图片",
                str(workspace_manager.root),
                "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All Files (*)",
            )
        if not image_path:
            return
        path = Path(image_path).expanduser()
        if not path.is_absolute():
            path = workspace_manager.root / path
        if not path.is_file() or not self._is_image_path(str(path)):
            QMessageBox.warning(self, "无法显示图片", f"不是受支持的图片文件：\n{path}")
            return
        self.messages.append(
            {
                "role": "assistant",
                "content": f"已打开图片：{path.name}",
                "_image_paths": [str(path.resolve())],
                "_show_images": True,
            }
        )
        self._render_messages()
        self.statusBar().showMessage(f"正在对话区显示图片：{path.name}")

    def send_message(self) -> None:
        text = self.composer.toPlainText().strip()
        if not text or (self.worker and self.worker.isRunning()):
            return
        if not self.ensure_authenticated():
            return
        if self._TEXT_MINING_TRIGGER.search(text) and not self._ensure_text_mining_model():
            self.statusBar().showMessage("未设置文本挖掘模型，已取消本次任务")
            return
        if hasattr(self, "_mark_active_session"):
            self._mark_active_session(text)
        self.composer.clear()
        self.messages.append({"role": "user", "content": text})
        self.current_assistant = ""
        from academic_agent.agent.session import session_store

        artifacts_before = list(session_store.get(self.session_id).artifacts)
        self.messages.append({
            "role": "assistant",
            "content": "Agent 正在分析…",
            "_image_paths": [],
            "_show_images": bool(self._IMAGE_REQUEST_TRIGGER.search(text)),
            "_execution_steps": [],
            "_execution_logs": [],
            "_artifacts_before": artifacts_before,
            "_artifact_paths": [],
        })
        self._render_messages()
        self.send_btn.setEnabled(False)
        self.statusBar().showMessage("Agent 正在思考…")
        model_messages = [
            {"role": message.get("role", "user"), "content": message.get("content", "")}
            for message in self.messages[:-1]
        ]
        work_mode = getattr(self, "agent_mode", "work") == "work"
        from academic_agent.views.workers import StreamWorker

        self.worker = StreamWorker(
            model_messages,
            self.user_id,
            self.session_id,
            str(workspace_manager.root) if work_mode else None,
            chat_only=not work_mode,
        )
        self.worker.chunk.connect(self.on_chunk)
        self.worker.progress.connect(self.on_progress)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, event: dict) -> None:
        """把 Planner 和 Tool Executor 的真实事件转换为用户可见步骤。"""
        if not self.messages or self.messages[-1].get("role") != "assistant":
            return
        steps = self.messages[-1].setdefault("_execution_steps", [])
        logs = self.messages[-1].setdefault("_execution_logs", [])
        event_type = str(event.get("type", ""))

        def finish_active() -> None:
            for step in steps:
                if step.get("status") == "running":
                    step["status"] = "completed"

        if event_type == "phase_started":
            key = str(event.get("key", "phase"))
            if not any(step.get("key") == key for step in steps):
                steps.append({
                    "key": key,
                    "label": str(event.get("label", "准备任务")),
                    "status": "running",
                })
        elif event_type == "plan_created":
            finish_active()
            for index, plan_step in enumerate(event.get("steps") or [], 1):
                description = str(plan_step.get("description", "执行任务步骤"))
                if any(step.get("label") == description for step in steps):
                    continue
                steps.append({
                    "key": f"plan_{index}",
                    "label": description,
                    "status": "pending",
                    "suggested_tools": list(plan_step.get("suggested_tools") or []),
                })
        elif event_type == "tool_started":
            tool = str(event.get("tool", "tool"))
            label = str(event.get("label") or "分析工具")
            steps.append({
                "key": f"tool_{tool}_{len(steps)}",
                "tool": tool,
                "label": f"调用工具：{label}",
                "status": "running",
            })
        elif event_type in {"tool_finished", "tool_failed", "tool_denied"}:
            tool = str(event.get("tool", ""))
            matched = next(
                (
                    step for step in reversed(steps)
                    if step.get("tool") == tool and step.get("status") == "running"
                ),
                None,
            )
            if matched is not None:
                success = event_type == "tool_finished" and bool(event.get("success", False))
                matched["status"] = "completed" if success else "failed"
        elif event_type == "tool_log":
            message = str(event.get("message", "")).strip()
            if message:
                logs.append({
                    "message": message,
                    "level": str(event.get("level", "info")),
                })
        elif event_type == "response_started":
            for step in steps:
                if step.get("status") in {"pending", "running"}:
                    step["status"] = "completed"
            if not any(step.get("key") == "response" for step in steps):
                steps.append({
                    "key": "response",
                    "label": "整理分析结果并生成回答",
                    "status": "running",
                })
        elif event_type == "completed":
            for step in steps:
                if step.get("status") in {"pending", "running"}:
                    step["status"] = "completed"
        self._schedule_stream_render()

    def on_chunk(self, content: str) -> None:
        if content == self.current_assistant:
            return
        # 保留原始响应中的 image_path/image_url；对话正文仍由 display_content
        # 负责做状态化展示，避免文件内容泄露到对话区。
        self.current_assistant = content
        self.messages[-1]["content"] = content
        if self.messages[-1].get("_show_images"):
            image_paths = self.messages[-1].setdefault("_image_paths", [])
            for image_path in self._extract_image_paths(content):
                if image_path not in image_paths:
                    image_paths.append(image_path)
        # 流式响应通常每个 token 都会触发一次信号。不要每次都重建整个
        # QTextBrowser 文档，否则 Qt 会不断重新计算高度并把视图顶来顶去。
        self._schedule_stream_render()

    def _schedule_stream_render(self) -> None:
        """将高频 token 更新合并为约 25 FPS 的界面刷新。"""
        from PyQt6.QtCore import QTimer

        timer = getattr(self, "_stream_render_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_stream_render)
            self._stream_render_timer = timer
        if not timer.isActive():
            timer.start(40)

    def _flush_stream_render(self) -> None:
        """刷新一次完整消息，但保持用户当前的阅读位置。"""
        self._render_messages()

    def _render_messages(self) -> None:
        parts = []
        primary = self.accent_color.darker(145).name()
        for msg in self.messages:
            role_key = msg.get("role", "assistant")
            role = "你" if role_key == "user" else "Agent"
            raw_content = msg.get("content", "")
            content = (
                self._display_agent_content(raw_content)
                if role_key != "user"
                else raw_content
            )
            body = (
                self._render_agent_body(content)
                if role_key != "user"
                else html.escape(content).replace("\n", "<br>")
            )
            # 图片路径只能从原始 Agent 响应中提取，不能从状态文本中提取。
            image_html = (
                self._render_image_results(raw_content, msg.get("_image_paths", []))
                if role_key != "user" and msg.get("_show_images", False) else ""
            )
            artifact_html = (
                self._render_artifact_results(msg.get("_artifact_paths", []))
                if role_key != "user" else ""
            )
            steps_html = (
                self._render_execution_steps(msg.get("_execution_steps", []))
                if role_key != "user" else ""
            )
            logs_html = (
                self._render_execution_logs(msg.get("_execution_logs", []))
                if role_key != "user" else ""
            )
            if not body:
                continue
            if role_key == "user":
                parts.append(
                    f'<div style="text-align:right; margin:12px 12px 12px 28%;">'
                    f'<span style="color:{primary}; font-weight:600;">{role}</span><br>'
                    f'<span style="background:#f0faf3; border:1px solid #b9ddc3; '
                    f'padding:11px 15px; border-radius:16px;">{body}</span></div>'
                )
            else:
                parts.append(
                    f'<div style="text-align:left; margin:12px 28% 12px 12%;">'
                    f'<span style="color:{primary}; font-weight:600;">{role}</span><br>'
                    f'<span style="background:#ffffff; border:1px solid #d8f0dd; '
                    f'padding:11px 15px; border-radius:14px;">'
                    f'{steps_html}{logs_html}{body}{image_html}{artifact_html}</span></div>'
                )
        scrollbar = self.chat_view.verticalScrollBar()
        previous_value = scrollbar.value()
        previous_maximum = scrollbar.maximum()
        was_at_bottom = previous_value >= max(0, previous_maximum - 12)
        self.chat_view.setUpdatesEnabled(False)
        try:
            self.chat_view.setHtml("".join(parts))
            if was_at_bottom:
                scrollbar.setValue(scrollbar.maximum())
            else:
                # 用户正在查看旧消息时，不要被流式输出强制拉到底部。
                scrollbar.setValue(min(previous_value, scrollbar.maximum()))
        finally:
            self.chat_view.setUpdatesEnabled(True)

    @staticmethod
    def _render_execution_steps(steps: list[dict]) -> str:
        if not steps:
            return ""
        icons = {
            "completed": ("✓", "#2e7d45"),
            "running": ("●", "#43a047"),
            "failed": ("×", "#b24b4b"),
            "pending": ("○", "#8aa492"),
        }
        lines = []
        for step in steps:
            icon, color = icons.get(str(step.get("status", "pending")), icons["pending"])
            label = html.escape(str(step.get("label", "执行步骤")))
            lines.append(
                f'<div style="margin:3px 0; color:#55745f;">'
                f'<span style="color:{color}; font-weight:700;">{icon}</span>&nbsp;{label}</div>'
            )
        return (
            '<div style="background:#f5faf6; border:1px solid #d8eadc; '
            'padding:10px 12px; margin:4px 0 12px;">'
            '<div style="color:#245c36; font-weight:700; margin-bottom:6px;">执行过程</div>'
            + "".join(lines)
            + '</div>'
        )

    @staticmethod
    def _render_execution_logs(logs: list[dict]) -> str:
        if not logs:
            return ""
        lines = []
        for entry in logs[-30:]:
            color = "#a33f3f" if entry.get("level") == "error" else "#4f6f59"
            message = html.escape(str(entry.get("message", "")))
            lines.append(
                f'<div style="margin:3px 0; color:{color}; font-family:Menlo,Monaco,monospace; '
                f'font-size:12px;">› {message}</div>'
            )
        return (
            '<div style="background:#fbfdfb; border:1px solid #e0ece3; '
            'padding:9px 12px; margin:0 0 12px;">'
            '<div style="color:#245c36; font-weight:700; margin-bottom:5px;">运行日志</div>'
            + "".join(lines)
            + '</div>'
        )

    @staticmethod
    def _sanitize_agent_text(content: str) -> str:
        """去除 Agent 输出中的 Markdown 星号装饰。"""
        return str(content).replace("*", "")

    def _display_agent_content(self, content: str) -> str:
        """对话区只展示文件操作状态，不展示工具读取到的文件原文。"""
        text = self._sanitize_agent_text(content)
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            try:
                payload = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text
        if not isinstance(payload, dict):
            return text

        file_name = payload.get("file") or payload.get("path") or "项目文件"
        image_target = (
            payload.get("image_path")
            or payload.get("image_url")
            or payload.get("visualization")
        )
        if image_target:
            return f"图表已生成：{Path(str(image_target)).name}"
        if "content" in payload and ("file" in payload or "path" in payload):
            return f"已读取文件：{file_name}（文件内容已隐藏，已交给 Agent 分析）"
        if "preview" in payload and payload.get("requires_confirmation"):
            operation_id = str(payload.get("operation_id", ""))
            if operation_id in self.confirmed_operations:
                return f"已完成文件操作：{payload.get('operation', 'generate')} → {file_name}"
            return f"已生成文件操作预览：{file_name}（等待用户确认）\n[[CONFIRM:{operation_id}]]"
        if isinstance(payload.get("results"), list):
            return f"已检索当前工作区文件：找到 {len(payload['results'])} 处匹配（详细内容已隐藏）"
        if isinstance(payload.get("files"), list):
            return f"已扫描当前工作区：发现 {len(payload['files'])} 个可操作文件"
        if isinstance(payload.get("artifacts"), list):
            return f"分析完成：已生成 {len(payload['artifacts'])} 个结果文件"
        if payload.get("success") and payload.get("operation"):
            return f"已完成文件操作：{payload.get('operation')} → {file_name}"
        if payload.get("error"):
            return f"文件操作失败：{payload['error']}"
        return text

    def _render_agent_body(self, content: str) -> str:
        """把待确认标记转换为对话区中的可点击确认链接。"""
        body = html.escape(content).replace("\n", "<br>")

        def replace_confirmation(match: re.Match[str]) -> str:
            operation_id = match.group(1)
            if operation_id in self.confirmed_operations:
                return '<br><span style="color:#4f805f; font-weight:600;">已确认执行</span>'
            return (
                f'<br><a href="agent-confirm://{operation_id}" '
                'style="color:#1769aa; font-weight:600; text-decoration:none;">'
                "点击确认执行</a>"
            )

        return re.sub(r"\[\[CONFIRM:([a-zA-Z0-9_-]+)\]\]", replace_confirmation, body)

    def _handle_chat_link(self, url) -> None:
        """处理对话区中的文件操作确认链接。"""
        if url.scheme() == "agent-file":
            self._open_artifact_link(url)
            return
        if url.scheme() != "agent-confirm":
            return
        operation_id = url.host() or url.path().lstrip("/")
        if not operation_id:
            return
        try:
            result = workspace_manager.confirm(operation_id)
            if not result.get("success"):
                raise RuntimeError(result.get("error", "文件操作确认失败"))
            self.confirmed_operations.add(operation_id)
            confirmed_file = str(result.get("file", ""))
            confirmed_path = Path(confirmed_file).expanduser()
            if not confirmed_path.is_absolute():
                confirmed_path = workspace_manager.root / confirmed_path
            if confirmed_path.is_file() and self.messages:
                artifact_paths = self.messages[-1].setdefault("_artifact_paths", [])
                resolved_file = str(confirmed_path.resolve())
                if resolved_file not in artifact_paths:
                    artifact_paths.append(resolved_file)
            if self._is_image_path(confirmed_file) and self.messages:
                self.messages[-1]["_show_images"] = True
                image_paths = self.messages[-1].setdefault("_image_paths", [])
                if confirmed_file not in image_paths:
                    image_paths.append(confirmed_file)
            self._render_messages()
            self._refresh_workspace_label()
            self.statusBar().showMessage(f"已确认并完成文件操作：{result.get('file', '')}")
        except Exception as exc:
            QMessageBox.warning(self, "文件操作失败", str(exc))

    @staticmethod
    def _artifact_token(path: Path) -> str:
        return base64.urlsafe_b64encode(str(path).encode("utf-8")).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_artifact_token(token: str) -> Path:
        padding = "=" * (-len(token) % 4)
        return Path(base64.urlsafe_b64decode(token + padding).decode("utf-8")).resolve()

    @staticmethod
    def _allowed_artifact_path(path: Path) -> bool:
        resolved = path.expanduser().resolve()
        return resolved.is_relative_to(output_root()) or resolved.is_relative_to(workspace_manager.root)

    def _open_artifact_link(self, url: QUrl) -> None:
        try:
            token = url.path().lstrip("/") or url.host()
            path = self._decode_artifact_token(token)
            if not self._allowed_artifact_path(path):
                raise ValueError("文件不在当前工作区或统一 output 目录中")
            if not path.is_file():
                raise FileNotFoundError(f"结果文件不存在：{path.name}")
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
                raise RuntimeError("系统没有可用于打开该文件的应用")
            self.statusBar().showMessage(f"已打开结果文件：{path.name}")
        except Exception as exc:
            QMessageBox.warning(self, "无法打开结果文件", str(exc))

    @staticmethod
    def _is_image_path(value: str) -> bool:
        return Path(str(value).split("?", 1)[0]).suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"
        }

    def _extract_image_paths(self, content: str) -> list[str]:
        """从工具中间结果、最终结果和代码产物中提取图片路径。"""
        candidates: list[str] = []

        def collect_images(value, allow_list_paths: bool = False) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, str) and (
                        key in {"image_path", "image_url", "visualization"}
                        or (key == "output_file" and self._is_image_path(item))
                    ):
                        candidates.append(item)
                    elif key == "artifacts":
                        # 只有工具明确返回的 artifacts 才允许从列表中提取图片；
                        # files/results 等普通检索列表中的图片不能进入对话区。
                        collect_images(item, allow_list_paths=True)
                    else:
                        collect_images(item, allow_list_paths=False)
            elif isinstance(value, list) and allow_list_paths:
                for item in value:
                    if isinstance(item, str) and self._is_image_path(item):
                        candidates.append(item)
                    else:
                        collect_images(item, allow_list_paths=True)

        cleaned = self._sanitize_agent_text(content)
        try:
            collect_images(json.loads(cleaned))
        except (TypeError, json.JSONDecodeError):
            try:
                collect_images(ast.literal_eval(cleaned))
            except (ValueError, SyntaxError):
                pass

        candidates.extend(re.findall(
            r"(?:image_path|image_url|output_file)\s*[=:]\s*[\"']([^\"']+\.(?:png|jpg|jpeg|webp|bmp|gif))[\"']",
            content,
            flags=re.IGNORECASE,
        ))
        # 仅支持明确的图片字段或 Markdown 图片语法，不扫描普通文本中的任意文件路径。
        candidates.extend(re.findall(
            r"!\[[^\]]*\]\(([^)\s]+\.(?:png|jpg|jpeg|webp|bmp|gif))(?:\s+[^)]*)?\)",
            content,
            flags=re.IGNORECASE,
        ))
        return list(dict.fromkeys(str(item) for item in candidates if item))

    def _render_image_results(self, content: str, extra_paths: list[str] | None = None) -> str:
        """把工具返回的图片路径转成 Qt 可渲染的本地图片。"""
        candidates = self._extract_image_paths(content)
        for item in extra_paths or []:
            if item not in candidates:
                candidates.append(item)
        rendered = []
        missing = []
        display_width = max(260, min(760, int(self.chat_view.viewport().width() * 0.68)))
        for candidate in candidates:
            if str(candidate).startswith(("http://", "https://")):
                continue
            if candidate.startswith("file://"):
                candidate = candidate.removeprefix("file://")
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = workspace_manager.root / path
            if path.is_file():
                uri = path.resolve().as_uri()
                rendered.append(
                    f'<br><img src="{html.escape(uri, quote=True)}" width="{display_width}">'
                    f'<br><span style="color:{self.accent_color.darker(120).name()}">'
                    f'图片：{html.escape(path.name)}</span>'
                )
            else:
                missing.append(str(candidate))
        if missing and not rendered:
            rendered.append(
                '<br><span style="color:#b26a00;">图片生成结果已返回，但当前界面暂未找到图片文件：'
                f'{html.escape(Path(missing[0]).name)}（请检查应用同级的 output 目录）</span>'
            )
        return "".join(rendered)

    def _render_artifact_results(self, paths: list[str] | None = None) -> str:
        """渲染经过路径校验的生成文件链接，不展示文件正文。"""
        links = []
        seen: set[str] = set()
        for value in paths or []:
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = workspace_manager.root / candidate
            candidate = candidate.resolve()
            key = str(candidate)
            if key in seen or not candidate.is_file() or not self._allowed_artifact_path(candidate):
                continue
            seen.add(key)
            token = self._artifact_token(candidate)
            suffix = candidate.suffix.lstrip(".").upper() or "FILE"
            links.append(
                f'<div style="margin:6px 0;">'
                f'<a href="agent-file://open/{token}" '
                'style="color:#1769aa; font-weight:600; text-decoration:none;">'
                f'打开文件：{html.escape(candidate.name)}</a>'
                f'&nbsp;<span style="color:#789080;">{html.escape(suffix)}</span></div>'
            )
        if not links:
            return ""
        return (
            '<div style="background:#f7fbf8; border:1px solid #d8eadc; '
            'padding:10px 12px; margin:12px 0 4px;">'
            '<div style="color:#245c36; font-weight:700; margin-bottom:5px;">结果文件</div>'
            + "".join(links)
            + '</div>'
        )

    def on_finished(self) -> None:
        timer = getattr(self, "_stream_render_timer", None)
        if timer is not None:
            timer.stop()
        if self.messages:
            from academic_agent.agent.session import session_store

            message = self.messages[-1]
            before = set(message.get("_artifacts_before", []))
            current = session_store.get(self.session_id).artifacts
            message["_artifact_paths"] = [
                path for path in current if path not in before
            ]
        self._render_messages()
        self.send_btn.setEnabled(True)
        if hasattr(self, "_load_history_sessions"):
            QTimer.singleShot(600, self._load_history_sessions)
        self.statusBar().showMessage("就绪")

    def on_failed(self, message: str) -> None:
        if self.messages:
            for step in self.messages[-1].get("_execution_steps", []):
                if step.get("status") == "running":
                    step["status"] = "failed"
        self.messages[-1]["content"] = f"❌ {message}"
        self._render_messages()
        self.send_btn.setEnabled(True)
        self.statusBar().showMessage("请求失败")
