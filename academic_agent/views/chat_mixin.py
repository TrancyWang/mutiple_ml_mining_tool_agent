"""对话、文件上传和 Agent 输出渲染。"""

from __future__ import annotations

import ast
import base64
import html
import json
import os
import re
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QPushButton

from academic_agent.infrastructure.workspace_manager import workspace_manager
from academic_agent.infrastructure.runtime_paths import output_root


class ChatMixin:
    """负责会话状态、上传文件、流式响应和对话区渲染。"""

    _TEXT_MINING_TRIGGER = re.compile(
        r"文本挖掘|文本聚类|聚类|情感分析|情感识别|关键词提取|主题词|文本预处理|分词|意图识别|响应识别|行为识别"
    )
    _TEXT_REPAIR_TRIGGER = re.compile(
        r"聚类修复|修复聚类|主题簇修复|情感分析修复|修复情感|情绪分析修复|"
        r"repair[_ -]?(?:cluster|clustering|sentiment)",
        re.IGNORECASE,
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
        composer = getattr(self, "composer", None)
        if composer is not None and obj is composer and event.type() == QEvent.Type.KeyPress:
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
        from academic_agent.infrastructure.data_profiler import profile_file
        from academic_agent.agent.session import session_store
        from academic_agent.tools.text_mining_tools import text_mining_tools

        path = self._choose_upload_file()
        if not path:
            return
        # 先完成文件选择，再在确实要导入时登录，避免选择器被登录流程打断。
        if not self.ensure_authenticated():
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

    def _choose_upload_file(self) -> str | None:
        """打开稳定的单文件选择器，避免原生选择器出现文件无法确认的问题。"""
        dialog = QFileDialog(self, "上传文件到当前对话")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        supported_filter = (
            "支持的文件 (*.csv *.tsv *.xlsx *.xls *.xlsm *.ods *.txt *.log *.md *.rst "
            "*.json *.jsonl *.ndjson *.parquet *.feather *.html *.htm *.pdf *.docx *.pptx "
            "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.mp3 *.wav *.mp4 *.mov *.avi)"
        )
        dialog.setNameFilters([
            supported_filter,
            "所有文件 (*)",
        ])
        dialog.selectNameFilter(supported_filter)
        dialog.resize(900, 600)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        selected = dialog.selectedFiles()
        return selected[0] if selected else None

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
        if (
            self._TEXT_MINING_TRIGGER.search(text)
            and not self._TEXT_REPAIR_TRIGGER.search(text)
            and not self._ensure_text_mining_model()
        ):
            self.statusBar().showMessage("未设置文本挖掘模型，已取消本次任务")
            return
        if hasattr(self, "_mark_active_session"):
            self._mark_active_session(text)
        self.composer.clear()
        self.messages.append({"role": "user", "content": text})
        self.current_assistant = ""
        # 发送前由输入区旁的两个常驻按钮决定本次任务的审批方式。
        self._agent_auto_mode = getattr(self, "_approval_mode", "manual") == "auto"
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
        previous_status = {
            str(step.get("key")): str(step.get("status", "pending"))
            for step in steps
            if step.get("key")
        }

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
            created_plan_steps = []
            for index, plan_step in enumerate(event.get("steps") or [], 1):
                description = str(plan_step.get("description", "执行任务步骤"))
                if any(step.get("label") == description for step in steps):
                    continue
                created_plan_steps.append({
                    "key": f"plan_{index}",
                    "label": description,
                    "status": "pending",
                    "suggested_tools": list(plan_step.get("suggested_tools") or []),
                })
            steps.extend(created_plan_steps)
            if created_plan_steps:
                created_plan_steps[0]["status"] = "running"
        elif event_type == "plan_review_fallback":
            reason = str(event.get("reason") or "规划模型未返回可识别结构")
            logs.append({
                "message": f"Plan Review 已采用本地安全计划继续：{reason}",
                "level": "warning",
            })
            self.statusBar().showMessage("Plan Review 格式未稳定返回，已采用本地安全计划继续")
        elif event_type == "plan_review_required":
            plan_payload = event.get("plan") or {}
            review_key = "replan_review" if plan_payload.get("is_replan") else "plan_review"
            matched_review = next((step for step in steps if step.get("key") == review_key), None)
            if matched_review is None:
                steps.append({
                    "key": review_key,
                    "label": "等待确认新的 Plan" if review_key == "replan_review" else "等待确认自然语言 Plan",
                    "status": "running",
                })
            else:
                matched_review["status"] = "running"
            self._show_plan_review(event.get("plan") or {}, event.get("board") or {})
        elif event_type == "clarification_required":
            info = event.get("information") or {}
            key = f"clarification_{info.get('info_id', len(steps))}"
            steps.append({
                "key": key,
                "label": f"补充信息：{info.get('topic', '规划信息')}",
                "status": "running",
            })
            self._show_clarification(info, key)
        elif event_type == "task_board_created":
            # plan_* 是规划阶段的静态预览；真正执行开始后改为展示动态任务板。
            # 重规划会重新从 T1 编号，因此必须清理上一轮 task_* 行，不能复用旧状态。
            if event.get("is_replan"):
                steps[:] = [
                    step for step in steps
                    if not str(step.get("key", "")).startswith("task_")
                ]
            steps[:] = [
                step for step in steps
                if not re.match(r"^plan_\d+$", str(step.get("key", "")))
            ]
            for task in (event.get("board") or {}).get("tasks", []):
                task_id = str(task.get("id", "task"))
                matched = next((step for step in steps if step.get("key") == f"task_{task_id}"), None)
                if matched is None:
                    matched = {
                        "key": f"task_{task_id}",
                        "label": str(task.get("title", "执行任务")),
                        "status": "pending",
                    }
                    steps.append(matched)
                matched["status"] = self._ui_task_status(task.get("status"))
        elif event_type == "task_started":
            task = event.get("task") or {}
            task_id = str(task.get("id", "task"))
            matched = next((step for step in steps if step.get("key") == f"task_{task_id}"), None)
            if matched is None:
                matched = {"key": f"task_{task_id}", "label": str(task.get("title", "执行任务"))}
                steps.append(matched)
            matched["status"] = "running"
        elif event_type == "task_checked":
            task_id = str(event.get("task_id", "task"))
            matched = next((step for step in steps if step.get("key") == f"task_{task_id}"), None)
            if matched is not None:
                matched["status"] = "running" if event.get("passed") else "failed"
            feedback = str(event.get("feedback", "")).strip()
            if feedback and not event.get("passed"):
                logs.append({"message": f"任务检查反馈：{feedback}", "level": "error"})
        elif event_type == "task_completed":
            task = event.get("task") or {}
            task_id = str(task.get("id", "task"))
            matched = next((step for step in steps if step.get("key") == f"task_{task_id}"), None)
            if matched is not None:
                matched["status"] = "completed"
        elif event_type == "task_retry":
            task = event.get("task") or {}
            task_id = str(task.get("id", "task"))
            matched = next((step for step in steps if step.get("key") == f"task_{task_id}"), None)
            if matched is not None:
                matched["status"] = "pending"
            logs.append({
                "message": f"任务将重试：{event.get('feedback', '')}",
                "level": "error",
            })
        elif event_type == "task_blocked":
            task = event.get("task") or {}
            task_id = str(task.get("id", "task"))
            matched = next((step for step in steps if step.get("key") == f"task_{task_id}"), None)
            if matched is not None:
                matched["status"] = "failed"
        elif event_type == "task_board_updated":
            for task in (event.get("board") or {}).get("tasks", []):
                task_id = str(task.get("id", "task"))
                if not any(step.get("key") == f"task_{task_id}" for step in steps):
                    steps.append({
                        "key": f"task_{task_id}",
                        "label": str(task.get("title", "新增任务")),
                        "status": self._ui_task_status(task.get("status")),
                    })
        elif event_type in {"task_board_state", "convergence_completed", "convergence_blocked"}:
            board = event.get("board") or {}
            for task in board.get("tasks", []):
                task_id = str(task.get("id", "task"))
                matched = next((step for step in steps if step.get("key") == f"task_{task_id}"), None)
                if matched is not None:
                    matched["status"] = self._ui_task_status(task.get("status"))
        elif event_type == "task_repair_started":
            logs.append({
                "message": f"已切换到规则修复路径：{event.get('repair_kind', '文本分析结果')}（不重新调用 BERT）",
                "level": "info",
            })
        elif event_type == "replan_requested":
            logs.append({
                "message": f"Reflection 建议重新规划：{event.get('reason', '')}",
                "level": "error",
            })
        elif event_type == "configuration_required":
            if not any(step.get("key") == "algorithm_configuration" for step in steps):
                steps.append({
                    "key": "algorithm_configuration",
                    "label": "等待确认算法与参数",
                    "status": "running",
                })
            self._show_algorithm_configuration(event.get("configuration") or {})
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
            if not any(step.get("key") == "response" for step in steps):
                steps.append({
                    "key": "response",
                    "label": "整理分析结果并生成回答",
                    "status": "running",
                })
        elif event_type == "completed":
            response = next((step for step in steps if step.get("key") == "response"), None)
            if response is not None:
                response["status"] = "completed"
        self._flash_newly_completed_steps(previous_status, steps)
        self._schedule_stream_render()

    @staticmethod
    def _ui_task_status(status: object) -> str:
        """把 TaskStatus 和界面图标状态解耦。"""
        value = str(status or "ready").lower()
        if value in {"completed", "complete"}:
            return "completed"
        if value in {"running", "verifying", "waiting_user"}:
            return "running"
        if value in {"failed", "blocked"}:
            return "failed"
        return "pending"

    def _flash_newly_completed_steps(
        self,
        previous_status: dict[str, str],
        steps: list[dict],
    ) -> None:
        """让每个刚完成的步骤短暂高亮，随后恢复为普通完成状态。"""
        now = time.monotonic()
        flashed = False
        for step in steps:
            key = str(step.get("key", ""))
            if step.get("status") == "completed" and previous_status.get(key) != "completed":
                step["_completion_flash_until"] = now + 0.72
                flashed = True
        if not flashed:
            return
        timer = getattr(self, "_execution_flash_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._clear_execution_flashes)
            self._execution_flash_timer = timer
        timer.start(720)

    def _clear_execution_flashes(self) -> None:
        now = time.monotonic()
        has_future_flash = False
        changed = False
        if self.messages:
            for step in self.messages[-1].get("_execution_steps", []):
                until = float(step.get("_completion_flash_until", 0) or 0)
                if until and until <= now:
                    step.pop("_completion_flash_until", None)
                    changed = True
                elif until > now:
                    has_future_flash = True
        if changed:
            self._render_messages()
        if has_future_flash:
            timer = getattr(self, "_execution_flash_timer", None)
            if timer is not None:
                timer.start(120)

    def _show_plan_review(self, plan: dict, board: dict) -> None:
        """在对话下方显示 Plan 卡片，避免模态窗口打断当前对话。"""
        self._pending_plan_review = {"plan": plan, "board": board}
        self._pending_plan_review_key = "replan_review" if plan.get("is_replan") else "plan_review"
        self._pending_clarification = None
        if getattr(self, "_agent_auto_mode", False):
            # 全自动模式不展示中间审批卡片，也不让后台线程在这里等待。
            self._finish_plan_approval(auto_mode=True)
            return
        if plan.get("is_replan"):
            self.agent_action_title.setText("任务遇到问题，需要重新规划")
            reason = str(plan.get("replan_reason") or "当前任务未通过检查")
            self.agent_action_context.setText(
                f"{reason}。Agent 已根据已有结果生成新的执行计划，请选择如何继续。"
            )
        else:
            self.agent_action_title.setText("先确认一下执行方向")
            self.agent_action_context.setText(
                "Agent 已经把你的需求整理成一份执行计划。这里展示的是行动安排，不是最终结果；确认后才会开始调用工具。"
            )
        preview = [str(plan.get("plan_document") or "未生成可读的 Plan 文书").strip()]
        tasks = board.get("tasks") or []
        if tasks:
            preview.append("\n执行步骤预览：")
            for index, task in enumerate(tasks, 1):
                title = str(task.get("title") or task.get("description") or "执行任务")
                goal = str(task.get("task_goal") or task.get("description") or "").strip()
                done_when = str(task.get("done_when") or "完成后由检查模块确认").strip()
                preview.append(f"{index}. {title}")
                if goal and goal != title:
                    preview.append(f"   目标：{goal}")
                preview.append(f"   完成标准：{done_when}")
        self.agent_action_details.setPlainText("\n".join(item for item in preview if item))
        self.agent_action_hint.setText(
            "当前执行方式为“请求批准”，请确认后继续；如果方向不对，请点击“取消这次任务”。"
        )
        self.agent_action_details.setVisible(True)
        self.agent_action_options.setVisible(False)
        self.agent_action_submit_btn.setText("确认执行计划")
        self.agent_action_submit_btn.setVisible(True)
        self.agent_action_submit_btn.setEnabled(True)
        self.agent_action_cancel_btn.setVisible(True)
        self.agent_action_panel.setVisible(True)
        self.agent_action_panel.raise_()
        self.statusBar().showMessage("请在对话下方确认 Agent 的执行计划")

    def _show_clarification(self, information: dict, key: str) -> None:
        """在对话下方解释信息缺口，让用户通过选项回答。"""
        self._pending_clarification = {"information": information, "key": key}
        self._pending_plan_review = None
        topic = str(information.get("topic") or "规划信息")
        question = str(information.get("question") or "请补充必要信息")
        if getattr(self, "_agent_auto_mode", False):
            # 全自动模式直接采用规划模型提供的首选项，不打断任务执行。
            self._answer_clarification_automatically(information, key)
            return
        why_needed = str(information.get("why_needed") or "")
        expected = str(
            information.get("expected_format")
            or information.get("answer_hint")
            or ""
        ).strip()
        example = str(information.get("example") or "").strip()
        self.agent_action_title.setText(f"需要你补充：{topic}")
        self.agent_action_context.setText(question)
        detail_lines = []
        if why_needed:
            detail_lines.append(f"为什么需要：{why_needed}")
        if expected:
            detail_lines.append(f"建议回答格式：{expected}")
        if example:
            detail_lines.append(f"示例：{example}")
        self.agent_action_details.setPlainText("\n".join(detail_lines))
        self.agent_action_hint.setText("请选择最符合你实际情况的一项；如果都不完全匹配，选择最接近的一项即可。")
        self.agent_action_details.setVisible(bool(detail_lines))
        self._populate_action_options(information)
        self.agent_action_submit_btn.setText("提交选择")
        self.agent_action_submit_btn.setVisible(True)
        self.agent_action_submit_btn.setEnabled(False)
        self.agent_action_cancel_btn.setVisible(True)
        self.agent_action_panel.setVisible(True)
        self.agent_action_panel.raise_()
        self.statusBar().showMessage("请在对话下方补充这项信息")

    def _populate_action_options(self, information: dict) -> None:
        """把规划模型提供的选项渲染成可点击按钮，不要求用户自由输入。"""
        while self.agent_action_options_layout.count():
            item = self.agent_action_options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        raw_options = information.get("options") or []
        options = []
        for item in raw_options[:6]:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("value") or "").strip()
                value = str(item.get("value") or label).strip()
                description = str(item.get("description") or "").strip()
            else:
                label = str(item).strip()
                value = label
                description = ""
            if label and value:
                options.append({"label": label, "value": value, "description": description})

        if not options:
            default = str(information.get("default_assumption") or "").strip()
            value = default or "未特别指定，请采用低风险、可逆的默认方案。"
            options.append({
                "label": "按建议的默认方案继续",
                "value": value,
                "description": default or "不额外指定特殊要求。",
            })

        self._pending_option_value = None
        self._pending_option_buttons = []
        for option in options:
            label = option["label"]
            description = option["description"]
            button = QPushButton(
                label if not description else f"{label}\n{description}",
                objectName="agent_action_option",
            )
            button.setCheckable(True)
            button.setToolTip(description or label)
            button.clicked.connect(
                lambda _checked=False, value=option["value"], selected=button:
                self._select_action_option(value, selected)
            )
            self._pending_option_buttons.append(button)
            self.agent_action_options_layout.addWidget(button)
        self.agent_action_options.setVisible(True)

    def _select_action_option(self, value: str, selected) -> None:
        self._pending_option_value = str(value)
        for button in getattr(self, "_pending_option_buttons", []):
            button.setChecked(button is selected)
        self.agent_action_submit_btn.setEnabled(True)
        self.statusBar().showMessage("已选择一项，点击“提交选择”继续")

    def _hide_agent_action_panel(self) -> None:
        self._pending_plan_review = None
        self._pending_plan_review_key = "plan_review"
        self._pending_clarification = None
        self.agent_action_panel.setVisible(False)

    def _submit_pending_action(self) -> None:
        """根据当前卡片类型提交人工确认或补充信息。"""
        if getattr(self, "_pending_plan_review", None):
            self._request_plan_approval()
        elif getattr(self, "_pending_clarification", None):
            self._submit_pending_clarification()

    def _approval_mode_changed(self, _index: int) -> None:
        """响应输入区的唯一执行方式选择器。"""
        selector = getattr(self, "approval_mode_selector", None)
        auto_mode = bool(selector is not None and selector.currentData() == "auto")
        self._set_approval_mode(auto_mode, announce=True)
        if auto_mode and getattr(self, "_pending_plan_review", None):
            self._finish_plan_approval(auto_mode=True)
        elif auto_mode and getattr(self, "_pending_clarification", None):
            pending = self._pending_clarification
            self._answer_clarification_automatically(
                pending.get("information") or {},
                str(pending.get("key") or ""),
            )

    def _set_approval_mode(self, auto_mode: bool, announce: bool = True) -> None:
        """更新唯一执行方式选择器。"""
        self._approval_mode = "auto" if auto_mode else "manual"
        self._agent_auto_mode = auto_mode
        selector = getattr(self, "approval_mode_selector", None)
        if selector is not None:
            selector.blockSignals(True)
            selector.setCurrentIndex(selector.findData("auto" if auto_mode else "manual"))
            selector.blockSignals(False)
        if announce:
            self.statusBar().showMessage(
                "已选择请求批准：Plan 生成后由你确认"
                if not auto_mode else
                "已选择帮我批准：本次任务将自动确认 Plan 并继续执行"
            )

    def _finish_plan_approval(self, auto_mode: bool) -> None:
        if not getattr(self, "_pending_plan_review", None):
            return
        review_key = getattr(self, "_pending_plan_review_key", "plan_review")
        for step in self.messages[-1].get("_execution_steps", []):
            if step.get("key") == review_key:
                step["status"] = "completed"
        self._hide_agent_action_panel()
        self.statusBar().showMessage(
            "已选择帮我批准，Agent 将全自动执行"
            if auto_mode else "已请求批准，正在生成任务清单"
        )
        if self.worker is not None:
            self.worker.set_plan_decision({"approved": True, "auto_mode": auto_mode})
        self._render_messages()

    def _request_plan_approval(self) -> None:
        self._set_approval_mode(False, announce=False)
        if getattr(self, "_pending_plan_review", None):
            self._finish_plan_approval(auto_mode=False)

    def _auto_approve_plan(self) -> None:
        self._set_approval_mode(True, announce=False)
        if getattr(self, "_pending_plan_review", None):
            self._finish_plan_approval(auto_mode=True)
        elif getattr(self, "_pending_clarification", None):
            pending = self._pending_clarification
            self._answer_clarification_automatically(
                pending.get("information") or {},
                str(pending.get("key") or ""),
            )

    def _answer_clarification_automatically(self, information: dict, key: str) -> None:
        """全自动模式下选择规划模型提供的第一项建议。"""
        options = information.get("options") or []
        answer = ""
        if options:
            first = options[0]
            answer = str(first.get("value") if isinstance(first, dict) else first).strip()
        if not answer:
            answer = str(information.get("default_assumption") or "").strip()
        if not answer:
            answer = "未特别指定，请采用低风险、可逆的默认方案。"
        for step in self.messages[-1].get("_execution_steps", []):
            if step.get("key") == key:
                step["status"] = "completed"
        self._hide_agent_action_panel()
        self.messages[-1].setdefault("_execution_logs", []).append({
            "message": f"全自动模式已选择：{answer}",
            "level": "info",
        })
        if self.worker is not None:
            self.worker.set_clarification_answer(answer)
        self.statusBar().showMessage("全自动模式：已自动选择规划选项，正在继续")
        self._render_messages()

    def _cancel_pending_action(self) -> None:
        if getattr(self, "_pending_plan_review", None):
            review_key = getattr(self, "_pending_plan_review_key", "plan_review")
            for step in self.messages[-1].get("_execution_steps", []):
                if step.get("key") == review_key:
                    step["status"] = "failed"
            self._hide_agent_action_panel()
            self.statusBar().showMessage("已取消 Plan 执行")
            if self.worker is not None:
                self.worker.set_plan_decision({"cancelled": True})
        elif getattr(self, "_pending_clarification", None):
            key = self._pending_clarification.get("key")
            for step in self.messages[-1].get("_execution_steps", []):
                if step.get("key") == key:
                    step["status"] = "failed"
            self._hide_agent_action_panel()
            self.statusBar().showMessage("未补充必要信息，已取消本次任务")
            if self.worker is not None:
                self.worker.set_clarification_answer(None)
        self._render_messages()

    def _submit_pending_clarification(self) -> None:
        pending = getattr(self, "_pending_clarification", None)
        if not pending:
            return
        answer = str(getattr(self, "_pending_option_value", "") or "").strip()
        if not answer:
            self.statusBar().showMessage("请先点选一项，再提交")
            return
        key = pending.get("key")
        for step in self.messages[-1].get("_execution_steps", []):
            if step.get("key") == key:
                step["status"] = "completed"
        self._hide_agent_action_panel()
        self.statusBar().showMessage("信息已提交，Agent 正在继续规划")
        if self.worker is not None:
            self.worker.set_clarification_answer(answer)
        self._render_messages()

    def _show_algorithm_configuration(self, configuration: dict) -> None:
        """在后台 Agent 等待时打开算法确认窗口，不阻塞 Qt 界面。"""
        if getattr(self, "_agent_auto_mode", False):
            selection = self._default_algorithm_configuration(configuration)
            self.statusBar().showMessage("全自动模式：已采用推荐算法和默认参数")
            for step in self.messages[-1].get("_execution_steps", []):
                if step.get("key") == "algorithm_configuration":
                    step["status"] = "completed"
            if self.worker is not None:
                self.worker.set_plan_configuration(selection)
            self._render_messages()
            return

        from academic_agent.views.dialogs import AlgorithmSelectionDialog

        dialog = AlgorithmSelectionDialog(configuration, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            selection = dialog.selected_configuration
            self.statusBar().showMessage(
                f"已确认：{selection.get('algorithm_label', '推荐算法')}，正在继续执行"
            )
            for step in self.messages[-1].get("_execution_steps", []):
                if step.get("key") == "algorithm_configuration":
                    step["status"] = "completed"
            if self.worker is not None:
                self.worker.set_plan_configuration(selection)
        else:
            for step in self.messages[-1].get("_execution_steps", []):
                if step.get("key") == "algorithm_configuration":
                    step["status"] = "failed"
            self.statusBar().showMessage("已取消算法配置")
            if self.worker is not None:
                self.worker.set_plan_configuration({"cancelled": True})
        self._render_messages()

    @staticmethod
    def _default_algorithm_configuration(configuration: dict) -> dict:
        """构造和算法确认对话框中“全自动化”选项一致的默认配置。"""
        from academic_agent.agent.planning.configuration import machine_learning_configuration

        active = dict(configuration or {})
        task_options = active.get("task_options") or []
        task_type = str(task_options[0].get("key")) if task_options else str(active.get("kind", ""))
        if task_options and task_type:
            active = machine_learning_configuration(task_type)
        options = active.get("options") or []
        default_key = active.get("default_algorithm")
        option = next(
            (item for item in options if item.get("key") == default_key),
            options[0] if options else {},
        )
        parameters = {
            str(spec["name"]): spec.get("default")
            for spec in option.get("parameters", [])
            if spec.get("name")
        }
        kind = str(active.get("kind", task_type))
        tool_arguments = dict(parameters)
        if kind == "text_clustering":
            tool_arguments["algorithm"] = option.get("key")
        elif kind == "sentiment_analysis":
            tool_arguments["mode"] = option.get("key")
        elif kind in {"classification", "regression"}:
            tool_arguments["model_type"] = option.get("key")
        elif kind == "causal_inference":
            tool_arguments["method"] = option.get("key")
        return {
            "kind": kind,
            "task_type": task_type or kind,
            "algorithm": option.get("key"),
            "algorithm_label": option.get("label"),
            "mode": "auto",
            "parameters": parameters,
            "tool_arguments": tool_arguments,
        }


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
        from PySide6.QtCore import QTimer

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
            flashing = float(step.get("_completion_flash_until", 0) or 0) > time.monotonic()
            row_style = (
                "margin:3px 0; padding:2px 6px; background:#e8f5e9; border-radius:6px;"
                if flashing else "margin:3px 0; padding:2px 6px;"
            )
            lines.append(
                f'<div style="{row_style} color:#55745f;">'
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
