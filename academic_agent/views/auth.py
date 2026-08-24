"""Qt6 登录与模型凭据配置。"""

from __future__ import annotations

from PyQt6.QtWidgets import (
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
            QLineEdit, QComboBox { background:#ffffff; color:#245c36; border:1px solid #bcd9c4; border-radius:12px; padding:10px 13px; min-height:30px; }
            QLineEdit:focus, QComboBox:focus { border:2px solid #43a047; }
            QCheckBox { color:#245c36; padding:6px 0; }
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
        self.username_edit.setPlaceholderText("用户名")
        self.username_edit.textChanged.connect(self._update_root_fields)
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("管理员密码；普通用户可留空")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Gemini", "gemini")
        self.provider_combo.addItem("千问（新加坡）", "qwen")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("请输入对应服务的 API Key")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("用户名", self.username_edit)
        form.addRow("密码", self.password_edit)
        form.addRow("模型服务", self.provider_combo)
        form.addRow("API Key", self.api_key_edit)
        layout.addLayout(form)

        self.remember_key = QCheckBox("记住 API Key（保存在本机 SQLite，下次自动填充）")
        self.remember_key.setChecked(True)
        layout.addWidget(self.remember_key)
        self.error_label = QLabel("", objectName="error")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox()
        login_button = buttons.addButton("登录", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("退出", QDialogButtonBox.ButtonRole.RejectRole)
        login_button.clicked.connect(self._submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._restore_last_account()
        self._update_root_fields(self.username_edit.text())

    def _update_root_fields(self, username: str) -> None:
        """管理员凭据来自 .env，不让 API Key 输入框造成必须填写的误解。"""
        is_root_name = str(username).strip() == ROOT_USERNAME
        self.provider_combo.setEnabled(not is_root_name)
        self.api_key_edit.setEnabled(not is_root_name)
        self.remember_key.setEnabled(not is_root_name)
        if is_root_name:
            self.api_key_edit.clear()
            self.api_key_edit.setPlaceholderText("管理员登录后自动读取应用内 .env")
        else:
            self.api_key_edit.setPlaceholderText("请输入对应服务的 API Key")

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
