"""侧栏常用功能组件。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QComboBox, QToolButton, QVBoxLayout, QWidget


class FeaturePanelMixin:
    """负责常用功能分组和自然语言提示词注入。"""

    def _add_feature_section(
        self,
        layout: QVBoxLayout,
        title: str,
        options: list[tuple[str, str]],
    ) -> None:
        layout.addWidget(QLabel(title, objectName="section_title"))
        combo = QComboBox(objectName="feature_select")
        combo.addItem("选择一项任务…")
        combo.setProperty("feature_section", title)
        combo.setToolTip(f"从“{title}”中选择一个常用任务，选中后会填入输入框")
        for label, prompt in options:
            combo.addItem(label, prompt)
        combo.currentIndexChanged.connect(
            lambda index, widget=combo: self._use_feature_prompt(widget, index)
        )
        layout.addWidget(combo)

    @staticmethod
    def _toggle_feature_group(toggle: QToolButton, panel: QWidget, checked: bool) -> None:
        panel.setVisible(checked)
        toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def _use_feature_prompt(self, combo: QComboBox, index: int) -> None:
        if index <= 0:
            return
        if not self.ensure_authenticated():
            combo.setCurrentIndex(0)
            return
        prompt = combo.itemData(index)
        if prompt == "__SHOW_IMAGE__":
            combo.setCurrentIndex(0)
            self.show_image_file()
            return
        if prompt:
            self.composer.setPlainText(str(prompt))
            self.composer.setFocus()
        combo.setCurrentIndex(0)
