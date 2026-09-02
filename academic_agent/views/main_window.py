"""Academic Agent PySide6 主窗口。"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer, Qt, QSize
from PySide6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QDialog,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from academic_agent.controllers import ApplicationController
from academic_agent.models import AgentMode, ApplicationState, AuthSession
from academic_agent.models.feature_catalog import FEATURE_SECTIONS
from academic_agent.views.auth import LoginDialog
from academic_agent.views.chat_mixin import ChatMixin
from academic_agent.views.feature_mixin import FeaturePanelMixin
from academic_agent.views.file_mixin import FilePanelMixin
from academic_agent.views.styles import BASE_STYLESHEET, build_stylesheet
from academic_agent.views.workspace_mixin import WorkspaceMixin
from academic_agent.views.model_mixin import ModelMixin
from academic_agent.infrastructure.runtime_paths import resource_root


class CodexChatWindow(
    ChatMixin,
    FilePanelMixin,
    WorkspaceMixin,
    FeaturePanelMixin,
    ModelMixin,
    QMainWindow,
):
    """只展示 Codex 风格的 PySide6 对话界面，不复刻原 Qt 项目的多 Tab 界面。"""

    @property
    def auth_session(self):
        return self.state.auth_session

    @auth_session.setter
    def auth_session(self, value) -> None:
        self.state.auth_session = value

    @property
    def user_id(self) -> str:
        return self.state.user_id

    @user_id.setter
    def user_id(self, value: str) -> None:
        self.state.user_id = str(value)

    @property
    def messages(self) -> list[dict]:
        return self.state.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self.state.messages = value

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self.state.session_id = str(value)

    @property
    def agent_mode(self) -> str:
        return self.state.mode.value

    @agent_mode.setter
    def agent_mode(self, value: str) -> None:
        self.state.mode = AgentMode.parse(value)

    @property
    def current_assistant(self) -> str:
        return self.state.current_assistant

    @current_assistant.setter
    def current_assistant(self, value: str) -> None:
        self.state.current_assistant = str(value)

    @property
    def confirmed_operations(self) -> set[str]:
        return self.state.confirmed_operations

    @property
    def model_provider(self) -> str:
        return self.state.model_provider

    @model_provider.setter
    def model_provider(self, value: str) -> None:
        self.state.model_provider = str(value)

    def __init__(self, auth_session: AuthSession | None = None, user_id: str = "qt_default_user"):
        super().__init__()
        configured_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower() or "auto"
        self.state = ApplicationState.create(
            auth_session=auth_session,
            fallback_user_id=user_id,
            model_provider=configured_provider,
        )
        self.controller = ApplicationController.create(self.state)
        self.worker = None
        self.settings_window = None
        self.theme_settings = QSettings("AcademicAgent", "QtClient")
        self.model_settings = QSettings("AcademicAgent", "Models")
        icon_path = resource_root() / "resources" / "academic_agent_icon.png"
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        saved_model_root = str(self.model_settings.value("pretrained_models_dir", "") or "").strip()
        if saved_model_root and Path(saved_model_root).is_dir():
            os.environ["PRETRAINED_MODELS_DIR"] = saved_model_root
        saved_sentiment = str(self.model_settings.value("sentiment_model_path", "") or "").strip()
        if saved_sentiment and Path(saved_sentiment).is_dir():
            os.environ["SENTIMENT_MODEL_PATH"] = saved_sentiment
        saved_dictionary = str(self.model_settings.value("custom_dictionary_path", "") or "").strip()
        if saved_dictionary and Path(saved_dictionary).is_file():
            os.environ["CUSTOM_DICTIONARY_PATH"] = saved_dictionary
        # 某些控制器会在窗口构造前预加载工具模块，因此显式同步适配器状态，
        # 确保保存的情感模型和自定义词典立即生效。
        try:
            from academic_agent.integrations.video_text_adapter import source_video_text_adapter
            if saved_sentiment and Path(saved_sentiment).is_dir():
                source_video_text_adapter.set_sentiment_model_path(saved_sentiment)
            if saved_dictionary and Path(saved_dictionary).is_file():
                source_video_text_adapter.set_custom_dictionary(saved_dictionary)
        except Exception:
            pass
        saved_color = self.theme_settings.value("accent_color", "#43A047")
        self.accent_color = QColor(str(saved_color))
        if not self.accent_color.isValid():
            self.accent_color = QColor("#43A047")
        self._build_ui()
        self._load_project_list()
        self._load_history_sessions()
        self.file_refresh_timer = QTimer(self)
        self.file_refresh_timer.setInterval(2000)
        self.file_refresh_timer.timeout.connect(self._refresh_open_file_panel)
        self.file_refresh_timer.start()
        # 提前完成一次本地 Agent 初始化：打包版若缺少运行模块或配置异常，
        # 用户打开应用就能看到具体原因，不必等到第一轮对话才发现问题。
        QTimer.singleShot(0, self._initialize_agent)

    def ensure_authenticated(self, force: bool = False) -> bool:
        """首次使用 Agent/数据功能时登录；已有会话可直接继续。"""
        if self.auth_session is not None and not force:
            return True

        dialog = LoginDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.session is None:
            self.statusBar().showMessage("未登录，当前功能未执行")
            return False

        session = dialog.session
        env_file = self.controller.auth.apply(session)
        self.state.login(session)
        if hasattr(self, "chat_view"):
            self.chat_view.clear()
        self._active_history_item = None
        self._load_project_list()
        self._load_history_sessions()
        if env_file is None and session.is_root:
            QMessageBox.warning(
                self,
                "未找到 .env",
                "管理员登录成功，但没有找到真实 .env；请将 .env 放在 app 旁边后重启。",
            )
        self.controller.agent.reset()
        self._update_auth_status()
        self._initialize_agent()
        return True

    def open_login(self) -> None:
        """打开清晰可见的账户登录入口。"""
        self.ensure_authenticated(force=True)

    def switch_account(self) -> None:
        """主动切换登录用户。"""
        self.ensure_authenticated(force=True)

    def logout(self) -> None:
        """退出当前账户，只清理登录态，不删除历史对话和工作区文件。"""
        username = self.user_id
        if self.settings_window is not None:
            self.settings_window.close()
        self.controller.auth.logout(username)
        self.state.logout()
        self.chat_view.clear()
        self.controller.agent.reset()
        self._update_auth_status()
        self._load_project_list()
        self._load_history_sessions()
        self.statusBar().showMessage("已退出登录；历史对话和工作区文件仍然保留")

    def open_settings(self) -> None:
        """打开独立设置窗口。"""
        if not self.ensure_authenticated():
            return
        from academic_agent.views.settings_window import SettingsWindow

        if self.settings_window is None or not self.settings_window.isVisible():
            self.settings_window = SettingsWindow(self)
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def _update_auth_status(self) -> None:
        if self.auth_session is None:
            text = "未登录｜点击功能后登录"
            identity_text = "未登录"
            initials = "访"
        else:
            text = f"已登录：{self.auth_session.username}"
            initials = self.auth_session.username[:2].upper()
            identity_text = f"用户名：{self.auth_session.username}"
        if hasattr(self, "auth_status_label"):
            self.auth_status_label.setText(text)
        if hasattr(self, "account_btn"):
            self.account_btn.setVisible(self.auth_session is not None)
            self.account_btn.setText("账户")
            self.account_btn.setIcon(self._make_account_icon(initials))
            self.account_btn.setIconSize(QSize(26, 26))
            self._rebuild_account_menu()
        if hasattr(self, "login_btn"):
            self.login_btn.setVisible(self.auth_session is None)
        if hasattr(self, "account_identity_label"):
            self.account_identity_label.setText(identity_text)
            self.account_identity_label.setToolTip(text)

    def _make_account_icon(self, initials: str) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#dff2e4"))
        painter.drawEllipse(1, 1, 30, 30)
        painter.setPen(self.accent_color.darker(145))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, initials[:2])
        painter.end()
        return QIcon(pixmap)

    def _initialize_agent(self) -> None:
        """启动时初始化 Agent，并把底层异常显示在状态栏。"""
        _ready, message = self.controller.agent.initialize()
        self.statusBar().showMessage(message)
        self._update_auth_status()

    def _build_ui(self) -> None:
        from academic_agent.views.main_layout import build_main_window

        build_main_window(self)

    def _load_history_sessions(self) -> None:
        """从 SQLite 恢复最近的用户-Agent完整会话。"""
        if not hasattr(self, "recent_sessions"):
            return
        self.recent_sessions.clear()
        self._active_history_item = None
        if self.auth_session is None:
            self.recent_sessions.addItem("登录后加载历史会话")
            return

        sessions = self.controller.sessions.list_sessions(limit=20)

        if not sessions:
            self.recent_sessions.addItem("暂无历史会话")
            return
        for session in sessions:
            title = session["title"].replace("\n", " ").strip()
            display_title = title if len(title) <= 42 else title[:42].rstrip() + "…"
            item = QListWidgetItem(display_title)
            project_path = session.get("workspace_path", "")
            project_label = (
                "未关联项目"
                if session.get("is_unbound")
                else Path(project_path).name
            )
            if self.agent_mode == "work":
                item.setToolTip(
                    f"项目：{project_label}\n{title}\n"
                    f"{session['message_count']} 条消息\n点击加载完整会话"
                )
            else:
                item.setToolTip(f"{title}\n{session['message_count']} 条消息\n点击加载聊天")
            item.setData(Qt.ItemDataRole.UserRole, session)
            self.recent_sessions.addItem(item)
            if session.get("session_id") == self.session_id:
                self.recent_sessions.setCurrentItem(item)

    def _load_project_list(self) -> None:
        """Work 模式显示当前用户使用过的全部项目。"""
        if not hasattr(self, "sessions"):
            return
        self.sessions.clear()
        if self.auth_session is None:
            self.sessions.addItem("登录后加载项目")
            return
        current_path = str(self.controller.projects.current_root)
        projects = self.controller.projects.list_projects(limit=30)
        for project in projects:
            item = QListWidgetItem(str(project["name"]))
            item.setToolTip(
                f"{project['workspace_path']}\n{project.get('session_count', 0)} 个会话"
            )
            item.setData(Qt.ItemDataRole.UserRole, project)
            self.sessions.addItem(item)
            if project["workspace_path"] == current_path:
                self.sessions.setCurrentItem(item)

    def _select_project(self, item: QListWidgetItem) -> None:
        project = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(project, dict):
            return
        target = Path(str(project.get("workspace_path", ""))).expanduser()
        if not target.is_dir():
            QMessageBox.warning(self, "项目不可用", f"项目目录不存在：\n{target}")
            return
        if target.resolve() == self.controller.projects.current_root:
            return
        self._activate_workspace(target)

    def _show_project_context_menu(self, position) -> None:
        item = self.sessions.itemAt(position)
        if item is None or not isinstance(item.data(Qt.ItemDataRole.UserRole), dict):
            return
        self.sessions.setCurrentItem(item)
        menu = QMenu(self.sessions)
        delete_action = menu.addAction("永久删除项目…")
        delete_action.triggered.connect(self._delete_selected_project)
        menu.exec(self.sessions.mapToGlobal(position))

    def _show_session_context_menu(self, position) -> None:
        item = self.recent_sessions.itemAt(position)
        if item is None or not isinstance(item.data(Qt.ItemDataRole.UserRole), dict):
            return
        self.recent_sessions.setCurrentItem(item)
        menu = QMenu(self.recent_sessions)
        delete_action = menu.addAction("永久删除对话")
        delete_action.triggered.connect(self._delete_selected_session)
        menu.exec(self.recent_sessions.mapToGlobal(position))

    def _delete_selected_session(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "请稍候", "当前对话仍在运行，请结束后再删除。")
            return
        item = self.recent_sessions.currentItem()
        session = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not isinstance(session, dict):
            return
        title = str(session.get("title") or item.text()).replace("\n", " ").strip()
        choice = QMessageBox.warning(
            self,
            "永久删除对话",
            f"将永久删除这段对话及其全部消息和记忆：\n\n{title}\n\n此操作无法撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        try:
            active = str(session.get("session_id") or "") == self.session_id
            deleted = self.controller.sessions.delete_session(session)
            if active:
                self.chat_view.clear()
                self._active_history_item = None
            self._load_history_sessions()
            self.statusBar().showMessage(f"对话已永久删除，共清理 {deleted} 条消息")
        except Exception as exc:
            QMessageBox.critical(self, "删除对话失败", str(exc))

    def _delete_selected_project(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "请稍候", "当前任务仍在运行，请结束后再删除项目。")
            return
        item = self.sessions.currentItem()
        project = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not isinstance(project, dict):
            return
        name = str(project.get("name") or Path(str(project.get("workspace_path", ""))).name)
        path = str(project.get("workspace_path") or "")
        typed, accepted = QInputDialog.getText(
            self,
            "从 Agent 移除项目",
            f"将从 Academic Agent 中移除项目记录及关联对话：\n{path}\n\n"
            "不会删除该目录及其中的任何物理文件。\n"
            f"请输入项目名称“{name}”确认：",
        )
        if not accepted or typed.strip() != name:
            if accepted:
                QMessageBox.warning(self, "未删除", "项目名称不匹配，已取消永久删除。")
            return
        try:
            result = self.controller.projects.delete(project)
            self.chat_view.clear()
            self._active_history_item = None
            self._refresh_workspace_label()
            self._load_project_list()
            self._load_history_sessions()
            self.statusBar().showMessage(
                f"已从 Agent 移除项目：{name}；清理 {result.get('message_count', 0)} 条关联消息，物理文件未删除"
            )
        except Exception as exc:
            QMessageBox.critical(self, "删除项目失败", str(exc))

    def _switch_agent_mode(self, _index: int) -> None:
        if not hasattr(self, "mode_selector"):
            return
        changed = self.controller.set_mode(self.mode_selector.currentData() or "work")
        if changed:
            self.chat_view.clear()
            self._active_history_item = None
        self._set_mode_visibility()
        self._load_project_list()
        self._load_history_sessions()
        if self.agent_mode == "chat":
            self.statusBar().showMessage("Chat 模式：只显示当前用户的聊天历史，不操作项目文件")
        else:
            self.statusBar().showMessage("Work 模式：可选择项目并使用项目文件工具")

    def _set_mode_visibility(self) -> None:
        """切换 Work/Chat 两种界面范围。"""
        work_mode = getattr(self, "agent_mode", "work") == "work"
        if hasattr(self, "project_section"):
            self.project_section.setVisible(work_mode)
        if hasattr(self, "common_toggle"):
            self.common_toggle.setVisible(work_mode)
        if hasattr(self, "common_scroll"):
            self.common_scroll.setVisible(work_mode)
        if hasattr(self, "right_sidebar"):
            self.right_sidebar.setVisible(work_mode)
        if hasattr(self, "open_file_btn"):
            self.open_file_btn.setVisible(work_mode)
        if hasattr(self, "upload_btn"):
            # 上传文件属于当前会话，Chat 模式同样可以上传资料。
            self.upload_btn.setVisible(True)

    def _rebuild_account_menu(self) -> None:
        """构建左下角账户菜单，避免设置、主题和退出按钮挤在一起。"""
        if not hasattr(self, "account_btn"):
            return
        menu = self.account_btn.menu()
        if menu is None:
            menu = QMenu(self.account_btn)
            self.account_btn.setMenu(menu)
        menu.clear()
        if self.auth_session is None:
            menu.addAction("登录", lambda: self.ensure_authenticated(force=True))
        else:
            menu.addAction("设置", self.open_settings)
            menu.addSeparator()
            menu.addAction("退出登录", self.logout)

    def _add_current_session_entry(self, title: str = "新对话") -> None:
        """在历史区立即显示新 session，避免用户误以为新建按钮没有效果。"""
        if self.auth_session is None or not hasattr(self, "recent_sessions"):
            return
        item = QListWidgetItem(title)
        item.setToolTip("当前新会话；发送第一条消息后会保存到历史记录")
        item.setData(
            Qt.ItemDataRole.UserRole,
            {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "workspace_path": (
                    str(self.controller.projects.current_root)
                    if getattr(self, "agent_mode", "work") == "work"
                    else ""
                ),
                "is_unbound": getattr(self, "agent_mode", "work") != "work",
                "title": title,
                "message_count": 0,
                "transient": True,
            },
        )
        self.recent_sessions.insertItem(0, item)
        self.recent_sessions.setCurrentItem(item)
        self._active_history_item = item

    def _mark_active_session(self, title: str) -> None:
        """把当前 session 的首个问题即时显示为会话标题。"""
        item = getattr(self, "_active_history_item", None)
        if item is None or item.listWidget() is not self.recent_sessions:
            self._add_current_session_entry(title)
            item = getattr(self, "_active_history_item", None)
            if item is None:
                return
        clean_title = str(title).replace("\n", " ").strip()
        display_title = clean_title if len(clean_title) <= 42 else clean_title[:42].rstrip() + "…"
        item.setText(display_title)
        item.setToolTip(f"{clean_title}\n当前会话")
        data = item.data(Qt.ItemDataRole.UserRole) or {}
        data.update({
            "title": clean_title,
            "message_count": 1,
            "workspace_path": (
                str(self.controller.projects.current_root)
                if getattr(self, "agent_mode", "work") == "work"
                else ""
            ),
            "is_unbound": getattr(self, "agent_mode", "work") != "work",
        })
        item.setData(Qt.ItemDataRole.UserRole, data)

    def _load_history_session(self, item: QListWidgetItem) -> None:
        """点击历史记录，恢复该用户-Agent session 的全部消息。"""
        session = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(session, dict):
            return
        workspace_path = session.get("workspace_path") or None
        if (
            self.agent_mode == "work"
            and workspace_path
            and Path(workspace_path).is_dir()
            and Path(workspace_path).resolve() != self.controller.projects.current_root
        ):
            self.controller.projects.activate(workspace_path)
            self._refresh_workspace_label()
            self._load_project_list()
        messages = self.controller.sessions.load_session(session)
        if not messages:
            self.statusBar().showMessage("该历史会话暂无可恢复消息")
            return
        self._active_history_item = item
        self._render_messages()
        self.statusBar().showMessage(f"已加载历史会话：{session.get('title', '新会话')}")

    def switch_model_provider(self, provider: str, model_name: str | None = None) -> None:
        """切换当前 Qt 会话使用的模型服务。"""
        if not self.ensure_authenticated():
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "请稍候", "当前对话仍在进行，请完成后再切换模型。")
            return

        labels = {
            "auto": "自动",
            "gemini": "Gemini",
            "qwen": "Qwen 新加坡",
            "qwen-beijing": "Qwen 北京",
            "ollama": "Ollama 本地",
        }
        try:
            config = self.controller.agent.switch_provider(provider, model_name)
            active_provider = config.get("provider", provider)
            if active_provider == "ollama":
                self.model_btn.setText(f"模型：Ollama {config['model']}")
            else:
                self.model_btn.setText(f"模型：{labels.get(active_provider, active_provider)}")
            if provider == "ollama":
                self.statusBar().showMessage(
                    f"已切换到 Ollama：{config['model']}；请确保 Ollama 正在运行。"
                )
            else:
                self.statusBar().showMessage(f"已切换到 {labels.get(active_provider, active_provider)}")
        except Exception as exc:
            QMessageBox.warning(self, "模型切换失败", str(exc))

    def _build_feature_sections(self, layout: QVBoxLayout) -> None:
        for title, options in FEATURE_SECTIONS:
            self._add_feature_section(layout, title, list(options))

    def _apply_theme(self) -> None:
        self.setStyleSheet(build_stylesheet(self.base_stylesheet, self.accent_color))

    def reset_green_theme(self) -> None:
        self.accent_color = QColor("#43A047")
        self.theme_settings.setValue("accent_color", self.accent_color.name())
        self._apply_theme()
        self.statusBar().showMessage("已恢复绿色白色主题")

    def customize_green_theme(self) -> None:
        from PySide6.QtWidgets import QColorDialog

        color = QColorDialog.getColor(self.accent_color, self, "选择绿色主色")
        if not color.isValid():
            return
        if color.green() <= color.red() or color.green() <= color.blue() or color.lightness() < 45:
            QMessageBox.warning(self, "颜色不符合要求", "请选择明亮的绿色，不能使用黑色或非绿色颜色。")
            return
        self.accent_color = color
        self.theme_settings.setValue("accent_color", color.name())
        self._apply_theme()
        self.statusBar().showMessage(f"主题颜色已设置为 {color.name()}")
