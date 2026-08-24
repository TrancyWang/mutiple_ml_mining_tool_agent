"""右侧工作区文件浏览面板。"""

from __future__ import annotations

from pathlib import Path
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from academic_agent.infrastructure.workspace_manager import workspace_manager


class FilePanelMixin:
    """管理文件树、筛选、打开、预览和定时刷新。"""

    def _build_file_panel(self) -> QFrame:
        panel = QFrame(objectName="file_panel")
        panel.setMinimumWidth(330)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)

        header = QHBoxLayout()
        header.addWidget(QLabel("打开文件", objectName="conversation_title"))
        header.addStretch()
        close_btn = QPushButton("×", objectName="icon_button")
        close_btn.setFixedSize(36, 36)
        close_btn.setToolTip("关闭文件面板")
        close_btn.clicked.connect(self.toggle_file_panel)
        header.addWidget(close_btn)
        panel_layout.addLayout(header)

        self.workspace_path_label = QLabel(
            f"工作区 · {workspace_manager.root.name}", objectName="connection"
        )
        self.workspace_path_label.setToolTip(str(workspace_manager.root))
        panel_layout.addWidget(self.workspace_path_label)
        self.file_filter = QLineEdit(objectName="file_filter")
        self.file_filter.setPlaceholderText("筛选文件…")
        self.file_filter.textChanged.connect(self._refresh_file_tree)
        panel_layout.addWidget(self.file_filter)

        self.file_tree = QTreeWidget(objectName="file_tree")
        self.file_tree.setHeaderHidden(True)
        self.file_tree.itemClicked.connect(self._open_workspace_file)
        panel_layout.addWidget(self.file_tree, 2)

        self.file_preview = QPlainTextEdit(objectName="file_preview")
        self.file_preview.setReadOnly(True)
        self.file_preview.setPlaceholderText("从当前工作区选择文件查看内容")
        self.current_open_file: str | None = None
        self.current_open_file_mtime: float = 0.0
        self._file_tree_last_refresh = 0.0
        panel_layout.addWidget(self.file_preview, 3)
        self._refresh_file_tree()
        return panel

    def toggle_file_panel(self) -> None:
        if not self.ensure_authenticated():
            return
        self.file_panel_open = not self.file_panel_open
        self.file_panel.setVisible(self.file_panel_open)
        if self.file_panel_open:
            self._refresh_file_tree()
            self.right_content_splitter.setSizes([280, 440])
            current = self.main_splitter.sizes()
            if len(current) == 3 and current[2] < 300:
                self.main_splitter.setSizes([260, 680, 340])

    def _refresh_file_tree(self) -> None:
        if not hasattr(self, "file_tree"):
            return
        query = self.file_filter.text().strip().lower()
        self.file_tree.clear()
        nodes: dict[str, QTreeWidgetItem] = {}
        for relative in workspace_manager.list_files():
            if query and query not in relative.lower():
                continue
            parent = self.file_tree.invisibleRootItem()
            current_parts: list[str] = []
            for index, part in enumerate(Path(relative).parts):
                current_parts.append(part)
                key = "/".join(current_parts)
                item = nodes.get(key)
                if item is None:
                    item = QTreeWidgetItem(parent, [part])
                    nodes[key] = item
                parent = item
                if index == len(Path(relative).parts) - 1:
                    item.setData(0, Qt.ItemDataRole.UserRole, relative)
        self.file_tree.expandToDepth(0)
        self._file_tree_last_refresh = time.monotonic()

    def _refresh_open_file_panel(self) -> None:
        """定时同步工作区文件树，并刷新当前打开文件的内容。"""
        if not getattr(self, "file_panel_open", False):
            return
        self._refresh_workspace_label()
        self.workspace_path_label.setText(f"工作区 · {workspace_manager.root.name}")
        self.workspace_path_label.setToolTip(str(workspace_manager.root))
        if time.monotonic() - getattr(self, "_file_tree_last_refresh", 0.0) >= 10.0:
            self._refresh_file_tree()
        if not self.current_open_file:
            return
        target = workspace_manager.root / self.current_open_file
        if not target.is_file():
            self.current_open_file = None
            self.current_open_file_mtime = 0.0
            self.file_preview.clear()
            return
        mtime = target.stat().st_mtime
        if mtime != self.current_open_file_mtime:
            self._load_open_file(self.current_open_file)

    def _open_workspace_file(self, item: QTreeWidgetItem, _column: int) -> None:
        relative = item.data(0, Qt.ItemDataRole.UserRole)
        if relative:
            self._load_open_file(str(relative))

    def _load_open_file(self, relative: str) -> None:
        try:
            if self._is_image_path(relative):
                self.file_preview.setPlainText(
                    f"图片已发送到对话区展示\n\n{Path(relative).name}"
                )
                self.current_open_file = relative
                self.current_open_file_mtime = (workspace_manager.root / relative).stat().st_mtime
                self.show_image_file(str(workspace_manager.root / relative))
                return
            result = workspace_manager.read(relative)
            self.file_preview.setPlainText(result["content"])
            self.current_open_file = relative
            self.current_open_file_mtime = (workspace_manager.root / relative).stat().st_mtime
            self.statusBar().showMessage(f"已打开：{relative}")
        except Exception as exc:
            self.file_preview.setPlainText(f"无法打开文件：{exc}")
