"""主窗口布局构建器。保持 Qt 组件创建与窗口生命周期逻辑分离。"""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QKeySequence, QPainter, QPen, QPixmap, QFont, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QMenu,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QStackedWidget, QStatusBar, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from academic_agent.views.styles import BASE_STYLESHEET


def build_main_window(window) -> None:
    self = window
    self.setWindowTitle("Academic Multimodal Text Mining Agent")
    self.resize(1280, 820)
    self.setMinimumSize(900, 620)
    self.setFont(QFont("PingFang SC", 13))
    self.base_stylesheet = BASE_STYLESHEET

    root = QWidget()
    self.setCentralWidget(root)
    root_layout = QHBoxLayout(root)
    root_layout.setContentsMargins(0, 0, 0, 0)
    splitter = QSplitter()
    self.main_splitter = splitter
    root_layout.addWidget(splitter)

    sidebar = QFrame(objectName="sidebar")
    sidebar.setMinimumWidth(270)
    side_layout = QVBoxLayout(sidebar)
    side_layout.setContentsMargins(12, 14, 12, 12)
    side_layout.setSpacing(8)
    self.mode_selector = QComboBox(objectName="mode_selector")
    self.mode_selector.addItem("Academic Agent · Work", "work")
    self.mode_selector.addItem("Academic Agent · Chat", "chat")
    self.mode_selector.setToolTip("Work：选择项目并使用工作区；Chat：只保留聊天历史")
    self.mode_selector.currentIndexChanged.connect(self._switch_agent_mode)
    side_layout.addWidget(self.mode_selector)
    
    self.new_chat_btn = QPushButton("＋  新对话", objectName="new_chat")
    self.new_chat_btn.clicked.connect(self.new_chat)
    side_layout.addWidget(self.new_chat_btn)
    self.project_section = QWidget()
    project_layout = QVBoxLayout(self.project_section)
    project_layout.setContentsMargins(0, 0, 0, 0)
    project_layout.setSpacing(2)
    project_layout.addWidget(QLabel("项目", objectName="section_title"))
    self.sessions = QListWidget()
    self.sessions.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.sessions.setTextElideMode(Qt.TextElideMode.ElideRight)
    self.sessions.itemClicked.connect(self._select_project)
    self.sessions.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.sessions.customContextMenuRequested.connect(self._show_project_context_menu)
    self.sessions.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    self.sessions.setMinimumHeight(62)
    self.sessions.setMaximumHeight(150)
    project_layout.addWidget(self.sessions)

    workspace_bar = QHBoxLayout()
    self.workspace_label = QLabel(objectName="connection")
    self.workspace_label.setWordWrap(True)
    workspace_bar.addWidget(self.workspace_label, 1)
    self.workspace_btn = QToolButton(objectName="workspace_button")
    self.workspace_btn.setText("管理  ···")
    self.workspace_btn.setToolTip("选择已有工作区或新建工作区")
    workspace_menu = QMenu(self.workspace_btn)
    workspace_menu.addAction("选择已有工作区", self.select_workspace)
    workspace_menu.addAction("新建工作区", self.create_workspace)
    workspace_menu.addSeparator()
    workspace_menu.addAction("设置文件操作权限", self.configure_workspace_permissions)
    self.workspace_btn.setMenu(workspace_menu)
    self.workspace_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    self.workspace_btn.setArrowType(Qt.ArrowType.NoArrow)
    workspace_bar.addWidget(self.workspace_btn)
    project_layout.addLayout(workspace_bar)
    side_layout.addWidget(self.project_section)
    self._refresh_workspace_label()

    self.recent_title = QLabel("最近对话", objectName="section_title")
    side_layout.addWidget(self.recent_title)
    self.recent_sessions = QListWidget()
    self.recent_sessions.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.recent_sessions.setTextElideMode(Qt.TextElideMode.ElideRight)
    self.recent_sessions.itemClicked.connect(self._load_history_session)
    self.recent_sessions.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.recent_sessions.customContextMenuRequested.connect(self._show_session_context_menu)
    self.recent_sessions.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    self.recent_sessions.setMinimumHeight(180)
    side_layout.addWidget(self.recent_sessions, 1)
    self.delete_project_shortcut = QShortcut(QKeySequence("Delete"), self.sessions)
    self.delete_project_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    self.delete_project_shortcut.activated.connect(self._delete_selected_project)
    self.delete_session_shortcut = QShortcut(QKeySequence("Delete"), self.recent_sessions)
    self.delete_session_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    self.delete_session_shortcut.activated.connect(self._delete_selected_session)

    self.auth_status_label = QLabel(objectName="connection")
    # 登录状态统一显示在左下角账户按钮中，避免“已登录：用户”和“用户：用户”重复出现。
    self.auth_status_label.setVisible(False)
    side_layout.addStretch(1)
    account_bar = QFrame(objectName="account_bar")
    side_bottom = QHBoxLayout(account_bar)
    side_bottom.setContentsMargins(0, 8, 0, 0)
    self.account_identity_label = QLabel(objectName="account_identity")
    self.account_identity_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    side_bottom.addWidget(self.account_identity_label, 1)
    self.login_btn = QPushButton("登录账户", objectName="login_button")
    self.login_btn.setToolTip("使用用户名和密码登录")
    self.login_btn.setMinimumHeight(38)
    self.login_btn.clicked.connect(self.open_login)
    side_bottom.addWidget(self.login_btn)
    self.account_btn = QToolButton(objectName="account_button")
    self.account_btn.setText("···")
    self.account_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    self.account_btn.setArrowType(Qt.ArrowType.NoArrow)
    self.account_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    self.account_btn.setMinimumSize(78, 38)
    self.account_btn.setMaximumWidth(104)
    self.account_btn.setToolTip("账户菜单")
    side_bottom.addWidget(self.account_btn)
    side_layout.addWidget(account_bar)
    self._rebuild_account_menu()
    self._update_auth_status()
    self._set_mode_visibility()
    splitter.addWidget(sidebar)

    center = QWidget()
    center_layout = QVBoxLayout(center)
    center_layout.setContentsMargins(0, 0, 0, 0)
    topbar = QFrame(objectName="topbar")
    topbar.setMinimumHeight(56)
    topbar_layout = QHBoxLayout(topbar)
    topbar_layout.setContentsMargins(16, 0, 16, 0)
    topbar_layout.addWidget(QLabel("学术多模态 Agent", objectName="conversation_title"))
    topbar_layout.addStretch()

    self.open_file_btn = QToolButton(objectName="open_file")
    folder_pixmap = QPixmap(22, 22)
    folder_pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(folder_pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(self.accent_color.darker(135), 1.8))
    painter.drawLine(3, 7, 8, 7)
    painter.drawLine(8, 7, 10, 9)
    painter.drawRoundedRect(2, 7, 18, 12, 3, 3)
    painter.end()
    self.open_file_btn.setIcon(QIcon(folder_pixmap))
    self.open_file_btn.setIconSize(QSize(20, 20))
    self.open_file_btn.setAutoRaise(True)
    self.open_file_btn.setAccessibleName("打开文件")
    self.open_file_btn.setToolTip("打开文件面板")
    self.open_file_btn.clicked.connect(self.toggle_file_panel)
    topbar_layout.addWidget(self.open_file_btn)
    center_layout.addWidget(topbar)

    self.chat_view = QTextBrowser(objectName="chat")
    self.chat_view.setOpenExternalLinks(False)
    self.chat_view.setOpenLinks(False)
    self.chat_view.anchorClicked.connect(self._handle_chat_link)
    center_layout.addWidget(self.chat_view, 1)

    # 计划确认和关键信息补充都以内嵌操作卡片呈现，不用模态对话框打断对话。
    self.agent_action_panel = QFrame(objectName="agent_action_panel")
    self.agent_action_panel.setVisible(False)
    self.agent_action_panel.setSizePolicy(
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Maximum,
    )
    action_layout = QVBoxLayout(self.agent_action_panel)
    action_layout.setContentsMargins(14, 11, 14, 11)
    action_layout.setSpacing(7)
    self.agent_action_title = QLabel(objectName="agent_action_title")
    action_layout.addWidget(self.agent_action_title)
    self.agent_action_context = QLabel(objectName="agent_action_context")
    self.agent_action_context.setWordWrap(True)
    action_layout.addWidget(self.agent_action_context)
    self.agent_action_details = QPlainTextEdit(objectName="agent_action_details")
    self.agent_action_details.setReadOnly(True)
    self.agent_action_details.setMaximumHeight(170)
    action_layout.addWidget(self.agent_action_details)
    self.agent_action_hint = QLabel(objectName="agent_action_hint")
    self.agent_action_hint.setWordWrap(True)
    action_layout.addWidget(self.agent_action_hint)
    self.agent_action_options = QWidget(objectName="agent_action_options")
    self.agent_action_options_layout = QVBoxLayout(self.agent_action_options)
    self.agent_action_options_layout.setContentsMargins(0, 0, 0, 0)
    self.agent_action_options_layout.setSpacing(5)
    self.agent_action_options.setVisible(False)
    action_layout.addWidget(self.agent_action_options)
    action_buttons = QHBoxLayout()
    action_buttons.addStretch()
    self.agent_action_cancel_btn = QPushButton("取消这次任务", objectName="agent_action_cancel")
    self.agent_action_submit_btn = QPushButton("提交信息", objectName="agent_action_submit")
    self.agent_action_cancel_btn.clicked.connect(self._cancel_pending_action)
    self.agent_action_submit_btn.clicked.connect(self._submit_pending_action)
    action_buttons.addWidget(self.agent_action_cancel_btn)
    action_buttons.addWidget(self.agent_action_submit_btn)
    action_layout.addLayout(action_buttons)
    center_layout.addWidget(self.agent_action_panel)

    composer_box = QFrame(objectName="composer_box")
    composer_layout = QVBoxLayout(composer_box)
    composer_layout.setContentsMargins(8, 6, 8, 7)
    self.upload_btn = QPushButton("上传文件", objectName="upload_button")
    self.upload_btn.setToolTip("上传数据或文档到当前对话")
    self.upload_btn.setAccessibleName("上传文件")
    self.upload_btn.setMinimumWidth(96)
    self.upload_btn.clicked.connect(self.upload_file)
    self.composer = QPlainTextEdit(objectName="composer")
    self.composer.setPlaceholderText("给 Academic Agent 发送消息…")
    self.composer.setFixedHeight(78)
    self.composer.installEventFilter(self)
    composer_layout.addWidget(self.composer)
    composer_bar = QHBoxLayout()
    self.send_btn = QPushButton("发送", objectName="send")
    self.send_btn.clicked.connect(self.send_message)
    composer_bar.addWidget(self.upload_btn)
    model_labels = {
        "auto": "自动",
        "gemini": "Gemini",
        "qwen": "Qwen 新加坡",
        "qwen-beijing": "Qwen 北京",
        "ollama": "Ollama 本地",
    }
    self.model_btn = QPushButton(
        f"模型：{model_labels.get(self.model_provider, '自动')}",
        objectName="model_button",
    )
    self.model_btn.setToolTip("切换对话模型；自动模式按 Gemini → Qwen → Ollama 选择")
    model_menu = QMenu(self.model_btn)
    model_menu.addAction(
        "自动（Gemini → Qwen → Ollama）",
        lambda checked=False: self.switch_model_provider("auto"),
    )
    model_menu.addAction(
        "Gemini",
        lambda checked=False: self.switch_model_provider("gemini"),
    )
    model_menu.addAction(
        "Qwen 新加坡",
        lambda checked=False: self.switch_model_provider("qwen"),
    )
    model_menu.addAction(
        "Qwen 北京",
        lambda checked=False: self.switch_model_provider("qwen-beijing"),
    )
    model_menu.addAction(
        "Ollama：qwen3.5:2b（推荐工具调用）",
        lambda checked=False: self.switch_model_provider("ollama", "qwen3.5:2b"),
    )
    model_menu.addAction(
        "Ollama：qwen3.5:0.8b（工具调用）",
        lambda checked=False: self.switch_model_provider("ollama", "qwen3.5:0.8b"),
    )
    model_menu.addSeparator()
    model_menu.addSection("本地模型路径")
    model_menu.addAction("设置文本挖掘模型库…", self.configure_model_directory)
    model_menu.addAction("恢复默认模型目录", self.reset_model_directory)
    self.model_btn.setMenu(model_menu)
    composer_bar.addWidget(self.model_btn)

    # Plan 审批属于当前输入动作，固定放在模型按钮旁边，使用一个下拉框选择执行方式。
    # 选择只决定“是否需要人工确认”；人工模式下，具体节点仍由下方卡片提供确认动作。
    self._approval_mode = "manual"
    self._agent_auto_mode = False
    self.approval_mode_label = QLabel("执行方式", objectName="approval_mode_label")
    self.approval_mode_label.setToolTip("选择本次任务是否需要在关键节点停下来确认")
    self.approval_mode_selector = QComboBox(objectName="approval_mode_selector")
    self.approval_mode_selector.addItem("请求批准", "manual")
    self.approval_mode_selector.addItem("帮我批准", "auto")
    self.approval_mode_selector.setCurrentIndex(0)
    self.approval_mode_selector.setToolTip(
        "请求批准：Plan、算法和关键信息节点由你确认；帮我批准：Agent 自动完成这些节点"
    )
    self.approval_mode_selector.currentIndexChanged.connect(self._approval_mode_changed)
    composer_bar.addWidget(self.approval_mode_label)
    composer_bar.addWidget(self.approval_mode_selector)
    composer_bar.addStretch()
    composer_bar.addWidget(self.send_btn)
    composer_layout.addLayout(composer_bar)
    composer_shell = QWidget(objectName="composer_shell")
    composer_shell_layout = QHBoxLayout(composer_shell)
    composer_shell_layout.setContentsMargins(52, 8, 52, 10)
    composer_shell_layout.addWidget(composer_box)
    center_layout.addWidget(composer_shell)
    splitter.addWidget(center)

    self.right_sidebar = QFrame(objectName="right_sidebar")
    self.right_sidebar.setMinimumWidth(340)
    self.right_sidebar.setMaximumWidth(480)
    right_layout = QVBoxLayout(self.right_sidebar)
    right_layout.setContentsMargins(12, 14, 12, 12)
    right_layout.setSpacing(8)

    self.right_content_stack = QStackedWidget()
    self.common_page = QWidget()
    common_page_layout = QVBoxLayout(self.common_page)
    common_page_layout.setContentsMargins(0, 0, 0, 0)
    common_page_layout.setSpacing(8)

    self.common_toggle = QToolButton()
    self.common_toggle.setObjectName("common_toggle")
    self.common_toggle.setText("常用功能")
    self.common_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    self.common_toggle.setArrowType(Qt.ArrowType.DownArrow)
    self.common_toggle.setCheckable(True)
    self.common_toggle.setChecked(True)
    self.common_toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    common_page_layout.addWidget(self.common_toggle)

    self.common_panel = QWidget()
    common_layout = QVBoxLayout(self.common_panel)
    common_layout.setContentsMargins(0, 0, 0, 0)
    common_layout.setSpacing(6)
    self.common_scroll = QScrollArea(objectName="common_scroll")
    self.common_scroll.setWidgetResizable(True)
    self.common_scroll.setFrameShape(QFrame.Shape.NoFrame)
    self.common_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    self.common_scroll.setWidget(self.common_panel)
    self.common_toggle.toggled.connect(
        lambda checked: self._toggle_feature_group(self.common_toggle, self.common_scroll, checked)
    )
    common_page_layout.addWidget(self.common_scroll, 1)
    self._build_feature_sections(common_layout)

    self.file_panel = self._build_file_panel()
    self.right_content_stack.addWidget(self.common_page)
    self.right_content_stack.addWidget(self.file_panel)
    self.right_content_stack.setCurrentWidget(self.common_page)
    right_layout.addWidget(self.right_content_stack, 1)
    splitter.addWidget(self.right_sidebar)
    self.file_panel_open = False
    splitter.setStretchFactor(1, 1)
    splitter.setStretchFactor(2, 0)
    splitter.setSizes([276, 664, 340])
    self.setStatusBar(QStatusBar())
    self.statusBar().setSizeGripEnabled(False)
    self.statusBar().showMessage("本地 Agent 已就绪")
    self._set_mode_visibility()
    self._apply_theme()
