"""客户端对话框。"""

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QSpinBox,
    QTextEdit,
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


class AlgorithmSelectionDialog(QDialog):
    """在计划执行前确认文本挖掘或机器学习的算法和参数。"""

    def __init__(self, configuration: dict, parent=None):
        super().__init__(parent)
        self.configuration = dict(configuration or {})
        self.selected_configuration: dict = {}
        self.setWindowTitle(self.configuration.get("title", "确认算法与参数"))
        self.setMinimumWidth(560)
        self.resize(620, 520)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(QLabel(self.configuration.get("description", "请确认本次分析使用的算法和参数。")))

        task_options = self.configuration.get("task_options", [])
        self.task_combo = None
        if task_options:
            task_group = QGroupBox("机器学习任务类型")
            task_layout = QFormLayout(task_group)
            self.task_combo = QComboBox()
            for item in task_options:
                self.task_combo.addItem(item["label"], item["key"])
            task_layout.addRow("任务类型", self.task_combo)
            layout.addWidget(task_group)
            self.task_combo.currentIndexChanged.connect(self._task_changed)

        self.automatic = QCheckBox("全自动化（使用推荐算法和默认参数）")
        self.automatic.setChecked(True)
        self.automatic.setToolTip("跳过手动配置，使用当前任务的推荐算法和默认参数。")
        layout.addWidget(self.automatic)

        algorithm_group = QGroupBox("算法选择")
        algorithm_layout = QFormLayout(algorithm_group)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.currentIndexChanged.connect(self._algorithm_changed)
        self.algorithm_description = QLabel()
        self.algorithm_description.setWordWrap(True)
        self.algorithm_description.setObjectName("hint")
        algorithm_layout.addRow("算法", self.algorithm_combo)
        algorithm_layout.addRow("说明", self.algorithm_description)
        layout.addWidget(algorithm_group)

        self.parameters_group = QGroupBox("参数设置")
        self.parameters_layout = QFormLayout(self.parameters_group)
        layout.addWidget(self.parameters_group)
        layout.addStretch()

        buttons = QDialogButtonBox()
        buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        continue_button = buttons.addButton("确认并执行", QDialogButtonBox.ButtonRole.AcceptRole)
        continue_button.clicked.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.automatic.toggled.connect(self._toggle_manual_controls)

        self._active_configuration = self.configuration
        if self.task_combo is not None:
            self._task_changed(self.task_combo.currentIndex())
        else:
            self._populate_algorithms()
        self._toggle_manual_controls(True)

    def _task_changed(self, _index: int) -> None:
        from academic_agent.agent.planning.configuration import machine_learning_configuration

        task = self.task_combo.currentData()
        if task:
            self._active_configuration = machine_learning_configuration(str(task))
            self._populate_algorithms()

    def _populate_algorithms(self) -> None:
        self.algorithm_combo.blockSignals(True)
        self.algorithm_combo.clear()
        options = self._active_configuration.get("options", [])
        for option in options:
            self.algorithm_combo.addItem(option.get("label", option.get("key", "算法")), option)
        default_key = self._active_configuration.get("default_algorithm")
        if default_key:
            for index, option in enumerate(options):
                if option.get("key") == default_key:
                    self.algorithm_combo.setCurrentIndex(index)
                    break
        self.algorithm_combo.blockSignals(False)
        self._algorithm_changed(self.algorithm_combo.currentIndex())

    def _algorithm_changed(self, _index: int) -> None:
        option = self.algorithm_combo.currentData() or {}
        self.algorithm_description.setText(option.get("description", ""))
        while self.parameters_layout.rowCount():
            self.parameters_layout.removeRow(0)
        self._parameter_widgets: dict[str, object] = {}
        for spec in option.get("parameters", []):
            kind = spec.get("type")
            if kind == "int":
                widget = QSpinBox()
                widget.setRange(int(spec.get("min", -1000000)), int(spec.get("max", 1000000)))
                widget.setValue(int(spec.get("default", 0)))
            else:
                widget = QDoubleSpinBox()
                widget.setRange(float(spec.get("min", -1000000.0)), float(spec.get("max", 1000000.0)))
                widget.setDecimals(3)
                widget.setSingleStep(float(spec.get("step", 0.05)))
                widget.setValue(float(spec.get("default", 0.0)))
            self._parameter_widgets[str(spec["name"])] = widget
            self.parameters_layout.addRow(str(spec.get("label", spec["name"])), widget)
        self.parameters_group.setVisible(bool(option.get("parameters")))
        self._toggle_manual_controls(self.automatic.isChecked())

    def _toggle_manual_controls(self, automatic: bool) -> None:
        enabled = not automatic
        self.algorithm_combo.setEnabled(enabled)
        self.parameters_group.setEnabled(enabled)

    def _accept_selection(self) -> None:
        option = self.algorithm_combo.currentData() or {}
        if not option:
            QMessageBox.warning(self, "缺少算法", "请选择一个算法后继续。")
            return
        automatic = self.automatic.isChecked()
        parameters = {
            name: (widget.value() if hasattr(widget, "value") else None)
            for name, widget in self._parameter_widgets.items()
        }
        if automatic:
            parameters = {
                str(spec["name"]): spec.get("default")
                for spec in option.get("parameters", [])
            }
        kind = str(self._active_configuration.get("kind", ""))
        tool_arguments = dict(parameters)
        if kind == "text_clustering":
            tool_arguments["algorithm"] = option.get("key")
        elif kind == "sentiment_analysis":
            tool_arguments["mode"] = option.get("key")
        elif kind in {"classification", "regression"}:
            tool_arguments["model_type"] = option.get("key")
        elif kind == "causal_inference":
            tool_arguments["method"] = option.get("key")
        self.selected_configuration = {
            "kind": kind,
            "task_type": self.task_combo.currentData() if self.task_combo else kind,
            "algorithm": option.get("key"),
            "algorithm_label": option.get("label"),
            "mode": "auto" if automatic else "manual",
            "parameters": parameters,
            "tool_arguments": tool_arguments,
        }
        self.accept()


class PlanReviewDialog(QDialog):
    """显示自然语言 Plan 和初始任务板，等待用户确认方向。"""

    def __init__(self, plan: dict, board: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("确认 Agent 执行计划")
        self.setMinimumWidth(700)
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"目标：{plan.get('objective', '当前任务')}"))
        outputs = "、".join(str(item) for item in plan.get("expected_outputs", []))
        if outputs:
            layout.addWidget(QLabel(f"预期产物：{outputs}"))

        document = QTextEdit()
        document.setReadOnly(True)
        document.setPlainText(str(plan.get("plan_document") or "未生成可读 Plan 文书"))
        layout.addWidget(document, 1)

        task_text = ["初始任务清单："]
        for index, task in enumerate(board.get("tasks", []), 1):
            task_text.append(f"{index}. {task.get('title', task.get('description', '任务'))}")
            task_text.append(f"   完成条件：{task.get('done_when', '由检查模块判断')}")
        tasks = QTextEdit()
        tasks.setReadOnly(True)
        tasks.setMaximumHeight(170)
        tasks.setPlainText("\n".join(task_text))
        layout.addWidget(tasks)

        buttons = QDialogButtonBox()
        buttons.addButton("取消执行", QDialogButtonBox.ButtonRole.RejectRole)
        approve = buttons.addButton("确认计划并执行", QDialogButtonBox.ButtonRole.AcceptRole)
        approve.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
