"""Academic Agent 简化设置窗口。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from academic_agent.infrastructure.workspace_manager import workspace_manager
from academic_agent.infrastructure.storage_config import (
    StorageConfig,
    current_storage_config,
    save_storage_config,
)
from academic_agent.views.styles import (
    BASE_STYLESHEET,
    THEME_OPTIONS,
    build_stylesheet,
    theme_preset,
)


class SettingsWindow(QMainWindow):
    """参考 Codex 设置页的简化版独立窗口。"""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("设置 · Academic Agent")
        self.resize(920, 640)
        self.setMinimumSize(760, 520)
        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        sidebar = QFrame(objectName="settings_sidebar")
        sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 18, 12, 16)
        back_btn = QPushButton("← 返回应用", objectName="settings_back")
        back_btn.clicked.connect(self.close)
        side_layout.addWidget(back_btn)
        side_layout.addWidget(QLabel("设置", objectName="settings_title"))

        self.nav = QListWidget(objectName="settings_nav")
        self.nav.addItems(["常规", "模型", "数据源", "外观", "账户"])
        self.nav.setCurrentRow(0)
        side_layout.addWidget(self.nav, 1)
        side_layout.addWidget(QLabel("Academic Agent", objectName="settings_hint"))
        root_layout.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._general_page())
        self.stack.addWidget(self._model_page())
        self.stack.addWidget(self._mysql_page())
        self.stack.addWidget(self._appearance_page())
        self.stack.addWidget(self._account_page())
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        content_scroll = QScrollArea(objectName="settings_content_scroll")
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        content_scroll.setWidget(self.stack)
        root_layout.addWidget(content_scroll, 1)

    def _page(self, title: str, subtitle: str = "") -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(42, 34, 42, 34)
        layout.setSpacing(16)
        layout.addWidget(QLabel(title, objectName="settings_page_title"))
        if subtitle:
            label = QLabel(subtitle, objectName="settings_page_subtitle")
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch()
        return page, layout

    @staticmethod
    def _card(title: str, description: str, control: QWidget) -> QFrame:
        card = QFrame(objectName="settings_card")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        text_layout.addWidget(QLabel(title, objectName="settings_card_title"))
        detail = QLabel(description, objectName="settings_card_description")
        detail.setWordWrap(True)
        text_layout.addWidget(detail)
        layout.addLayout(text_layout, 1)
        layout.addWidget(control)
        return card

    def _general_page(self) -> QWidget:
        page, layout = self._page("常规", "管理工作区权限和启动后的基本行为。")
        layout.insertWidget(
            layout.count() - 1,
            QLabel("权限", objectName="settings_section_title"),
        )
        permission = QComboBox(objectName="settings_combo")
        permission.addItem("只读（每次修改前确认）", "read_only")
        permission.addItem("自动允许生成和编辑", "accept_edits")
        permission.addItem("自动允许全部文件操作", "accept_all")
        index = permission.findData(workspace_manager.permission_mode)
        permission.setCurrentIndex(max(0, index))
        permission.currentIndexChanged.connect(
            lambda: self._set_permission(permission.currentData())
        )
        layout.insertWidget(
            layout.count() - 1,
            self._card(
                "工作区文件操作权限",
                "设置 Agent 生成、编辑和删除当前工作区文件时是否需要确认。",
                permission,
            ),
        )

        history = QCheckBox("启动时加载最近历史问题", objectName="settings_checkbox")
        history.setChecked(True)
        history.setEnabled(False)
        layout.insertWidget(
            layout.count() - 1,
            self._card(
                "历史问题",
                "已启用 SQLite 历史问题恢复，问题会显示在左侧最近对话区域。",
                history,
            ),
        )

        storage_title = QLabel("记忆存储", objectName="settings_section_title")
        layout.insertWidget(layout.count() - 1, storage_title)
        storage = current_storage_config()
        self.elasticsearch_checkbox = QCheckBox(
            "启用本地 Elasticsearch", objectName="settings_checkbox"
        )
        self.elasticsearch_checkbox.setChecked(storage.elasticsearch)
        layout.insertWidget(
            layout.count() - 1,
            self._card(
                "Elasticsearch",
                "启用本地 Elasticsearch 后，用于保存可检索的完整会话；未启用时不连接 9200 端口。",
                self.elasticsearch_checkbox,
            ),
        )
        self.milvus_checkbox = QCheckBox(
            "启用本地 Milvus Lite", objectName="settings_checkbox"
        )
        self.milvus_checkbox.setChecked(storage.milvus)
        layout.insertWidget(
            layout.count() - 1,
            self._card(
                "Milvus Lite",
                "启用本地 Milvus Lite 后，用于 BGE 向量语义记忆；需要本地模型和可写数据目录。",
                self.milvus_checkbox,
            ),
        )
        storage_apply = QPushButton("保存存储设置", objectName="settings_action")
        storage_apply.clicked.connect(self._apply_storage)
        layout.insertWidget(
            layout.count() - 1,
            self._card(
                "默认存储",
                "两个选项都未勾选时，仅使用 SQLite 文件保存历史会话和消息。",
                storage_apply,
            ),
        )
        return page

    def _mysql_page(self) -> QWidget:
        from academic_agent.infrastructure.mysql_datasource import (
            MySQLProfile,
            load_saved_mysql_profile,
        )

        saved = load_saved_mysql_profile() or MySQLProfile()
        page, layout = self._page(
            "数据源",
            "配置后，Work 模式中的 Text2SQL 才能访问 MySQL。密码只保存在当前运行会话，不写入设置文件。",
        )
        form_group = QGroupBox("MySQL 连接")
        form = QFormLayout(form_group)
        self.mysql_host = QLineEdit(saved.host)
        self.mysql_port = QSpinBox()
        self.mysql_port.setRange(1, 65535)
        self.mysql_port.setValue(int(saved.port))
        self.mysql_database = QLineEdit(saved.database)
        self.mysql_username = QLineEdit(saved.username)
        self.mysql_password = QLineEdit()
        self.mysql_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.mysql_password.setPlaceholderText("每次启动后重新输入")
        self.mysql_tables = QTextEdit()
        self.mysql_tables.setPlaceholderText("可选：每行一个授权表名；留空表示使用数据库账号可见的表")
        self.mysql_tables.setMaximumHeight(92)
        self.mysql_max_rows = QSpinBox()
        self.mysql_max_rows.setRange(1, 100000)
        self.mysql_max_rows.setValue(int(saved.max_rows))
        self.mysql_timeout = QSpinBox()
        self.mysql_timeout.setRange(1, 120)
        self.mysql_timeout.setValue(int(saved.connect_timeout))
        form.addRow("主机", self.mysql_host)
        form.addRow("端口", self.mysql_port)
        form.addRow("数据库", self.mysql_database)
        form.addRow("用户名", self.mysql_username)
        form.addRow("密码", self.mysql_password)
        form.addRow("授权表", self.mysql_tables)
        form.addRow("最大返回行数", self.mysql_max_rows)
        form.addRow("连接超时（秒）", self.mysql_timeout)
        layout.insertWidget(layout.count() - 1, form_group)

        hint = QLabel(
            "安全边界：只允许单条 SELECT；查询必须使用已配置数据库中的表；"
            "结果会强制限制最大行数。业务指标还需要通过 TEXT2SQL_ONTOLOGY_PATH 配置目录。"
        )
        hint.setWordWrap(True)
        layout.insertWidget(layout.count() - 1, hint)
        actions = QHBoxLayout()
        save_button = QPushButton("测试连接并保存", objectName="settings_action")
        save_button.clicked.connect(self._save_mysql)
        clear_button = QPushButton("清除当前连接", objectName="settings_action")
        clear_button.clicked.connect(self._clear_mysql)
        actions.addWidget(save_button)
        actions.addWidget(clear_button)
        actions.addStretch()
        action_host = QWidget()
        action_host.setLayout(actions)
        layout.insertWidget(layout.count() - 1, action_host)
        return page

    def _save_mysql(self) -> None:
        from academic_agent.infrastructure.mysql_datasource import (
            MySQLProfile,
            mysql_data_source,
            save_mysql_profile,
        )

        tables = tuple(
            item.strip()
            for item in self.mysql_tables.toPlainText().splitlines()
            if item.strip()
        )
        profile = MySQLProfile(
            host=self.mysql_host.text().strip(),
            port=int(self.mysql_port.value()),
            database=self.mysql_database.text().strip(),
            username=self.mysql_username.text().strip(),
            allowed_tables=tables,
            max_rows=int(self.mysql_max_rows.value()),
            connect_timeout=int(self.mysql_timeout.value()),
        )
        try:
            mysql_data_source.configure(profile, self.mysql_password.text())
            result = mysql_data_source.test_connection()
            if not result.get("success"):
                raise RuntimeError(result.get("error", "连接失败"))
            save_mysql_profile(profile)
            schema = mysql_data_source.inspect_schema()
            count = len(schema.get("tables", [])) if schema.get("success") else 0
            self.main_window.statusBar().showMessage(f"MySQL 已连接；可用表 {count} 个。")
        except Exception as exc:
            QMessageBox.warning(self, "MySQL 配置失败", str(exc))

    def _clear_mysql(self) -> None:
        from academic_agent.infrastructure.mysql_datasource import (
            clear_saved_mysql_profile,
            mysql_data_source,
        )

        mysql_data_source.clear()
        clear_saved_mysql_profile()
        self.mysql_password.clear()
        self.main_window.statusBar().showMessage("MySQL 数据源已清除。")

    def _model_page(self) -> QWidget:
        page, layout = self._page(
            "模型",
            "管理文本挖掘模型库和可选的中文情绪模型；对话模型仍可在输入框底部切换。",
        )
        def path_card(title: str, description: str, value: str, buttons: list[tuple[str, object]]) -> QFrame:
            host = QWidget()
            row = QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            path_label = QLabel(value, objectName="settings_value")
            path_label.setWordWrap(True)
            path_label.setMinimumWidth(260)
            row.addWidget(path_label, 1)
            for label, callback in buttons:
                button = QPushButton(label, objectName="settings_action")
                button.clicked.connect(callback)
                row.addWidget(button)
            return self._card(title, description, host)

        model_value = str(
            self.main_window.model_settings.value(
                "pretrained_models_dir", "未指定（使用默认模型库）"
            )
            or "未指定（使用默认模型库）"
        )
        layout.insertWidget(
            layout.count() - 1,
            path_card(
                "文本挖掘模型库根目录",
                "请选择 pretrain_models 文件夹。程序会自动查找 BGE、通用五分类和中文八分类模型。",
                model_value,
                [("设置模型库", self._configure_model_directory)],
            ),
        )

        sentiment_value = str(
            self.main_window.model_settings.value(
                "sentiment_model_path",
                "未单独指定（使用模型库中的中文八分类模型）",
            )
            or "未单独指定（使用模型库中的中文八分类模型）"
        )
        layout.insertWidget(
            layout.count() - 1,
            path_card(
                "中文八分类情绪模型（可选）",
                "仅用于中文情绪分析模式。通用五分类模型不在这里选择，而是由上方模型库自动查找。",
                sentiment_value,
                [
                    ("选择中文模型", self._configure_sentiment_model),
                    ("使用模型库默认", self._reset_sentiment_model),
                ],
            ),
        )

        dictionary_value = str(
            self.main_window.model_settings.value("custom_dictionary_path", "未启用")
            or "未启用"
        )
        layout.insertWidget(
            layout.count() - 1,
            path_card(
                "关键词自定义词典",
                "关键词抽取前加载用户词典，用于关键词识别和分词。",
                dictionary_value,
                [
                    ("选择词典", self._configure_custom_dictionary),
                    ("停用词典", self._reset_custom_dictionary),
                ],
            ),
        )
        return page

    def _appearance_page(self) -> QWidget:
        page, layout = self._page("外观", "支持四套清新浅色皮肤，切换会立即生效，不会启用暗黑模式。")
        color = QLabel(self.main_window.accent_color.name(), objectName="settings_value")
        self.theme_color_value = color
        layout.insertWidget(
            layout.count() - 1,
            self._card("当前主题色", "应用当前皮肤使用的主色，也会影响按钮和选中状态。", color),
        )
        theme_selector = QComboBox(objectName="settings_combo")
        for label, mode in THEME_OPTIONS:
            theme_selector.addItem(label, mode)
        current_index = theme_selector.findData(self.main_window.theme_mode)
        theme_selector.setCurrentIndex(max(0, current_index))
        theme_selector.currentIndexChanged.connect(
            lambda index: self._change_theme_mode(theme_selector.itemData(index))
        )
        self.theme_selector = theme_selector
        layout.insertWidget(
            layout.count() - 1,
            self._card("界面皮肤", "选择薄荷绿、天空蓝、淡雅薰衣草或晨光奶油黄。", theme_selector),
        )
        theme_btn = QPushButton("选择主题主色", objectName="settings_action")
        theme_btn.clicked.connect(self._customize_theme)
        layout.insertWidget(
            layout.count() - 1,
            self._card("自定义主题色", "主色会同步用于按钮、选中状态和对话区强调色。", theme_btn),
        )
        reset_btn = QPushButton("恢复默认", objectName="settings_action")
        reset_btn.clicked.connect(self._reset_theme)
        layout.insertWidget(
            layout.count() - 1,
            self._card("默认主题", "恢复 Academic Agent 默认浅绿色主题。", reset_btn),
        )
        return page

    def _account_page(self) -> QWidget:
        page, layout = self._page("账户", "查看当前登录状态或切换用户。")
        username = getattr(self.main_window.auth_session, "username", "未登录")
        current = QLabel(str(username), objectName="settings_value")
        layout.insertWidget(
            layout.count() - 1,
            self._card("当前用户", "用户凭据和登录状态由本地 SQLite 保存。", current),
        )
        switch_btn = QPushButton("切换账户", objectName="settings_action")
        switch_btn.clicked.connect(self._switch_account)
        layout.insertWidget(
            layout.count() - 1,
            self._card("账户登录", "切换后会重新初始化 Agent 配置。", switch_btn),
        )
        return page

    def _set_permission(self, mode: str) -> None:
        try:
            workspace_manager.set_permission_mode(str(mode))
            self.main_window.statusBar().showMessage(f"工作区权限已更新：{mode}")
        except Exception as exc:
            self.main_window.statusBar().showMessage(f"权限设置失败：{exc}")

    def _apply_storage(self) -> None:
        config = save_storage_config(
            StorageConfig(
                elasticsearch=self.elasticsearch_checkbox.isChecked(),
                milvus=self.milvus_checkbox.isChecked(),
            )
        )
        try:
            from academic_agent.infrastructure.persistence.hybrid_store import hybrid_memory

            hybrid_memory.reload()
            self.main_window.statusBar().showMessage(
                f"记忆存储已切换为：{config.label}；SQLite 始终保留为历史主存储。"
            )
        except Exception as exc:
            self.main_window.statusBar().showMessage(f"存储服务重载失败：{exc}")

    def _configure_model_directory(self) -> None:
        if self.main_window.configure_model_directory():
            self.close()

    def _configure_sentiment_model(self) -> None:
        if self.main_window.configure_sentiment_model():
            self.close()

    def _reset_sentiment_model(self) -> None:
        self.main_window.reset_sentiment_model()
        self.close()

    def _configure_custom_dictionary(self) -> None:
        if self.main_window.configure_custom_dictionary():
            self.close()

    def _reset_custom_dictionary(self) -> None:
        self.main_window.reset_custom_dictionary()
        self.close()

    def _customize_theme(self) -> None:
        self.main_window.customize_green_theme()
        self.theme_color_value.setText(self.main_window.accent_color.name())
        self._apply_theme()

    def _reset_theme(self) -> None:
        self.main_window.reset_green_theme()
        self.theme_color_value.setText(self.main_window.accent_color.name())
        if hasattr(self, "theme_selector"):
            index = self.theme_selector.findData(self.main_window.theme_mode)
            self.theme_selector.blockSignals(True)
            self.theme_selector.setCurrentIndex(max(0, index))
            self.theme_selector.blockSignals(False)
        self._apply_theme()

    def _change_theme_mode(self, mode: object) -> None:
        self.main_window.set_theme_mode(str(mode))
        self._apply_theme()

    def _switch_account(self) -> None:
        self.main_window.switch_account()
        self.close()

    def _apply_theme(self) -> None:
        settings_styles = """
            QFrame#settings_sidebar { background:#fbfdfb; border-right:1px solid #e0eee3; }
            QLabel#settings_title { color:#52745b; font-size:24px; font-weight:700; padding:16px 8px; }
            QLabel#settings_page_title { color:#52745b; font-size:28px; font-weight:700; padding:4px 0 12px; }
            QLabel#settings_page_subtitle, QLabel#settings_hint { color:#789783; font-size:14px; }
            QLabel#settings_section_title { color:#52745b; font-size:17px; font-weight:700; padding-top:12px; }
            QListWidget#settings_nav { background:transparent; color:#52745b; font-size:15px; }
            QListWidget#settings_nav::item { padding:12px 14px; margin:2px 0; border-radius:11px; }
            QListWidget#settings_nav::item:selected { background:#eef9f1; color:#52745b; font-weight:600; }
            QFrame#settings_card { background:#ffffff; border:1px solid #e0eee3; border-radius:16px; }
            QLabel#settings_card_title { color:#52745b; font-size:15px; font-weight:700; }
            QLabel#settings_card_description { color:#718a78; font-size:14px; }
            QLabel#settings_value { color:#52745b; font-size:13px; padding:8px; }
            QComboBox#settings_combo { min-width:190px; }
            QCheckBox#settings_checkbox { color:#52745b; padding:8px; }
            QPushButton#settings_back, QPushButton#settings_action { padding:9px 16px; border-radius:12px; }
            """
        if self.main_window.effective_theme_mode != "sage":
            preset = theme_preset(self.main_window.effective_theme_mode)
            settings_styles += f"""
            QFrame#settings_sidebar {{ background:{preset['sidebar']}; border-color:{preset['divider']}; }}
            QLabel#settings_title, QLabel#settings_page_title,
            QLabel#settings_section_title, QLabel#settings_value,
            QListWidget#settings_nav, QLabel#settings_card_title {{ color:{preset['primary']}; }}
            QLabel#settings_page_subtitle, QLabel#settings_hint,
            QLabel#settings_card_description {{ color:{preset['secondary']}; }}
            QListWidget#settings_nav::item:selected {{ background:{preset['selection']}; color:{preset['primary']}; }}
            QFrame#settings_card {{ background:{preset['surface']}; border-color:{preset['border']}; }}
            QCheckBox#settings_checkbox {{ color:{preset['primary']}; }}
            """
        self.setStyleSheet(
            build_stylesheet(
                BASE_STYLESHEET,
                self.main_window.accent_color,
                self.main_window.effective_theme_mode,
            )
            + settings_styles
        )
