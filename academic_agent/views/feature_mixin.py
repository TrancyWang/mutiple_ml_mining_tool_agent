"""侧栏常用功能组件。"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QComboBox, QToolButton, QVBoxLayout, QWidget


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
        combo.addItem(title)
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
        if combo.itemText(0) == "文本挖掘" and not self._ensure_text_mining_model():
            combo.setCurrentIndex(0)
            return
        if prompt:
            self.composer.setPlainText(str(prompt))
            self.composer.setFocus()
        combo.setCurrentIndex(0)
