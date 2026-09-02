"""PySide6 登录与模型凭据配置。"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from academic_agent.controllers.auth import (
    ROOT_PASSWORD,
    ROOT_USERNAME,
    AuthController,
)
from academic_agent.models.auth import AuthSession


auth_controller = AuthController()


def load_root_env():
    """兼容旧调用；认证规则由 Controller 管理。"""
    return auth_controller.load_root_env()


def saved_auth_session() -> AuthSession | None:
    return auth_controller.saved_session()


def apply_manual_provider(session: AuthSession) -> None:
    auth_controller.apply(session)


def apply_auth_session(session: AuthSession):
    return auth_controller.apply(session)


class LoginDialog(QDialog):
    """启动前认证；管理员使用 .env，普通用户使用手动 API Key。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: AuthSession | None = None
        self.setWindowTitle("Academic Agent 登录")
        self.setMinimumWidth(520)
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog, QWidget { background:#ffffff; color:#245c36; }
            QLabel#title { color:#176b38; font-size:26px; font-weight:700; padding:4px 0 2px; }
            QLabel#hint { color:#5a8968; font-size:13px; line-height:1.5; padding-bottom:6px; }
            QLabel#error { color:#b24b4b; padding:6px 0; }
            QLineEdit, QComboBox { background:#ffffff; color:#245c36; border:1px solid #bcd9c4; border-radius:12px; padding:10px 13px; min-height:30px; selection-background-color:#c8e6c9; selection-color:#1b5e20; }
            QLineEdit:focus, QComboBox:focus { border:2px solid #43a047; }
            QCheckBox { color:#245c36; padding:6px 0; }
            QLabel#mode_hint { color:#5a8968; font-size:12px; padding:2px 0; }
            QPushButton { background:#ffffff; color:#245c36; border:1px solid #bcd9c4; border-radius:12px; padding:9px 18px; min-height:24px; }
            QPushButton:hover { background:#e8f5e9; border-color:#43a047; }
            QPushButton[text="登录"] { background:#43a047; color:#ffffff; border-color:#388e3c; border-radius:17px; font-weight:600; }
            QPushButton[text="登录"]:hover { background:#388e3c; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 26, 30, 24)
        layout.setSpacing(14)
        title = QLabel("Academic Agent", objectName="title")
        layout.addWidget(title)
        layout.addWidget(
            QLabel(
                "登录后即可使用工作区和分析工具。管理员使用预配置模型服务，普通用户需填写自己的 API Key。",
                objectName="hint",
            )
        )

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        self.username_edit = QLineEdit()
        self.username_edit.setObjectName("username_edit")
        self.username_edit.setEnabled(True)
        self.username_edit.setReadOnly(False)
        self.username_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.username_edit.setPlaceholderText("用户名")
        self._configure_input(self.username_edit)
        self.username_edit.textChanged.connect(self._update_root_fields)
        self.password_edit = QLineEdit()
        self.password_edit.setObjectName("password_edit")
        self.password_edit.setEnabled(True)
        self.password_edit.setReadOnly(False)
        self.password_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.password_edit.setClearButtonEnabled(True)
        self.password_edit.setPlaceholderText("点击此处输入管理员密码；普通用户可留空")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._configure_input(self.password_edit)
        self.password_edit.returnPressed.connect(self._submit)
        self.show_password = QCheckBox("显示密码")
        self.show_password.setToolTip("临时显示密码，方便确认是否输入正确")
        self.show_password.toggled.connect(
            lambda visible: self.password_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
            )
        )
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Gemini", "gemini")
        self.provider_combo.addItem("千问（新加坡）", "qwen")
        self.provider_combo.addItem("千问（北京）", "qwen-beijing")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setObjectName("api_key_edit")
        self.api_key_edit.setEnabled(True)
        self.api_key_edit.setReadOnly(False)
        self.api_key_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.api_key_edit.setClearButtonEnabled(True)
        self.api_key_edit.setPlaceholderText("请输入对应服务的 API Key")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._configure_input(self.api_key_edit)
        form.addRow("用户名", self.username_edit)
        form.addRow("密码", self.password_edit)
        form.addRow("", self.show_password)
        form.addRow("模型服务", self.provider_combo)
        form.addRow("API Key", self.api_key_edit)
        layout.addLayout(form)

        self.mode_hint = QLabel(objectName="mode_hint")
        self.mode_hint.setWordWrap(True)
        layout.addWidget(self.mode_hint)

        self.remember_key = QCheckBox("记住 API Key（保存在本机 SQLite，下次自动填充）")
        self.remember_key.setChecked(True)
        layout.addWidget(self.remember_key)
        self.error_label = QLabel("", objectName="error")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox()
        login_button = buttons.addButton("登录", QDialogButtonBox.ButtonRole.AcceptRole)
        self.login_button = login_button
        self.login_button.setDefault(True)
        self.login_button.setAutoDefault(True)
        self.login_button.setMinimumWidth(110)
        buttons.addButton("退出", QDialogButtonBox.ButtonRole.RejectRole)
        login_button.clicked.connect(self._submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._restore_last_account()
        self._update_root_fields(self.username_edit.text())
        self.setTabOrder(self.username_edit, self.password_edit)
        self.setTabOrder(self.password_edit, self.provider_combo)
        self.setTabOrder(self.provider_combo, self.api_key_edit)
        # 部分平台会在模态窗口显示时把焦点交给按钮或下拉框；延迟到
        # event loop 后重新聚焦，保证用户打开登录框即可直接输入用户名。
        QTimer.singleShot(0, self._focus_username)

    def _focus_username(self) -> None:
        """让用户名输入框在窗口显示后获得焦点，并把光标放到文本末尾。"""
        if not self.isVisible():
            return
        self.username_edit.setEnabled(True)
        self.username_edit.setReadOnly(False)
        self.username_edit.setFocus(Qt.FocusReason.OtherFocusReason)
        # 不全选恢复的用户名：部分 macOS/Qt 主题在全选状态下会把插入光标
        # 绘制得非常不明显，用户会误以为输入框不能编辑。
        self.username_edit.setCursorPosition(len(self.username_edit.text()))

    @staticmethod
    def _configure_input(widget: QLineEdit) -> None:
        """统一保证登录输入框可编辑，并使用清晰可见的文字/光标配色。"""
        palette = widget.palette()
        dark_text = QColor("#245c36")
        light_base = QColor("#ffffff")
        selected_base = QColor("#c8e6c9")
        selected_text = QColor("#1b5e20")
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, QPalette.ColorRole.Text, dark_text)
            palette.setColor(group, QPalette.ColorRole.Base, light_base)
            palette.setColor(group, QPalette.ColorRole.Highlight, selected_base)
            palette.setColor(group, QPalette.ColorRole.HighlightedText, selected_text)
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#7a9d84"))
        widget.setPalette(palette)
        widget.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def _update_root_fields(self, username: str) -> None:
        """管理员凭据来自 .env，不让 API Key 输入框造成必须填写的误解。"""
        is_root_name = str(username).strip() == ROOT_USERNAME
        self.provider_combo.setEnabled(not is_root_name)
        self.api_key_edit.setEnabled(not is_root_name)
        self.remember_key.setEnabled(not is_root_name)
        if is_root_name:
            self.api_key_edit.clear()
            self.api_key_edit.setPlaceholderText("管理员登录后自动读取应用内 .env")
            self.mode_hint.setText("管理员模式：只需填写密码，模型配置会从应用内 .env 自动读取。")
        else:
            self.api_key_edit.setPlaceholderText("请输入对应服务的 API Key")
            self.mode_hint.setText("普通用户模式：密码可以留空，但必须填写所选模型服务的 API Key。")

    def _restore_last_account(self) -> None:
        account = auth_controller.last_account()
        if not account:
            return
        self.username_edit.setText(str(account.get("username", "")))
        provider = str(account.get("provider", "gemini"))
        index = self.provider_combo.findData(provider)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)
        if bool(account.get("remember_api_key")):
            self.api_key_edit.setText(str(account.get("api_key", "")))

    def _submit(self) -> None:
        try:
            self.session = auth_controller.authenticate(
                username=self.username_edit.text(),
                password=self.password_edit.text(),
                provider=str(self.provider_combo.currentData() or "gemini"),
                api_key=self.api_key_edit.text(),
                remember_api_key=self.remember_key.isChecked(),
            )
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()
