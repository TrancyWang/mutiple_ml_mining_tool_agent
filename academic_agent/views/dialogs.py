"""客户端对话框。"""

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from academic_agent.infrastructure.dependency_manager import install_dependencies


class DependencyDialog(QDialog):
    """让用户选择缺失的功能依赖，不自动安装。"""

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择安装缺失功能")
        self.resize(560, 420)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("检测到以下功能缺少依赖，请选择需要安装的功能："))
        self.checks = []
        for item in items:
            check = QCheckBox(f"{item.feature}：{item.package}\n    {item.description}")
            check.setChecked(True)
            self.checks.append((check, item))
            layout.addWidget(check)
        layout.addStretch()
        self.status = QLabel("")
        layout.addWidget(self.status)
        buttons = QDialogButtonBox()
        self.install_btn = buttons.addButton("安装选中依赖", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("暂不安装", QDialogButtonBox.ButtonRole.RejectRole)
        self.install_btn.clicked.connect(self.install_selected)
        layout.addWidget(buttons)

    def install_selected(self):
        selected = [item for check, item in self.checks if check.isChecked()]
        if not selected:
            self.status.setText("请至少选择一个功能，或点击暂不安装。")
            return
        self.install_btn.setEnabled(False)
        self.status.setText("正在安装，请稍候……")
        QApplication.processEvents()
        success, output = install_dependencies(selected)
        if success:
            self.status.setText("安装完成。请重启客户端使新库生效。")
            self.install_btn.setText("关闭")
            self.install_btn.setEnabled(True)
            self.install_btn.clicked.disconnect()
            self.install_btn.clicked.connect(self.accept)
        else:
            self.status.setText("安装失败，请查看终端输出或手动执行 pip install。")
            QMessageBox.warning(self, "依赖安装失败", output)
            self.install_btn.setEnabled(True)
