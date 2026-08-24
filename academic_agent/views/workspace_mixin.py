"""工作区选择和文件操作权限。"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from academic_agent.infrastructure.workspace_manager import WorkspaceManager, workspace_manager


class WorkspaceMixin:
    """负责工作区切换、创建、刷新和操作权限设置。"""

    def create_workspace(self) -> None:
        if not self.ensure_authenticated():
            return
        parent = QFileDialog.getExistingDirectory(
            self, "选择工作区位置", str(workspace_manager.root.parent)
        )
        if not parent:
            return
        name, accepted = QInputDialog.getText(self, "新建工作区", "工作区名称：")
        if not accepted or not name.strip():
            return
        target = Path(parent) / name.strip()
        try:
            target.mkdir(parents=True, exist_ok=True)
            target_files = WorkspaceManager(target).list_files()
            parent_files = WorkspaceManager(parent).list_files()
            if not target_files and parent_files:
                choice = QMessageBox.question(
                    self,
                    "工作区目录确认",
                    f"新建目录为空，但父目录已有 {len(parent_files)} 个文件。\n\n"
                    f"是否直接使用父目录作为工作区？\n{parent}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                self._activate_workspace(Path(parent) if choice == QMessageBox.StandardButton.Yes else target)
            else:
                self._activate_workspace(target)
        except Exception as exc:
            QMessageBox.critical(self, "工作区创建失败", str(exc))

    def select_workspace(self) -> None:
        if not self.ensure_authenticated():
            return
        target = QFileDialog.getExistingDirectory(
            self, "选择已有工作区", str(workspace_manager.root.parent)
        )
        if target:
            try:
                self._activate_workspace(Path(target))
            except Exception as exc:
                QMessageBox.critical(self, "工作区切换失败", str(exc))

    def _activate_workspace(self, target: Path) -> None:
        if hasattr(self, "controller"):
            target = self.controller.projects.activate(target)
        else:
            workspace_manager.set_root(target)
        self._refresh_workspace_label()
        # 项目列表由 SQLite 中的历史项目和当前工作区统一生成，避免手工追加
        # 的 QListWidgetItem 没有 UserRole 数据，点击后无法再次切换。
        if hasattr(self, "_load_project_list"):
            self._load_project_list()
        # 项目切换后不复用旧项目的活动会话，避免文件上下文和历史对话串项目。
        self.chat_view.clear()
        if not hasattr(self, "controller"):
            self.messages.clear()
            self.current_assistant = ""
            self.confirmed_operations.clear()
        if hasattr(self, "_active_history_item"):
            self._active_history_item = None
        self._load_history_sessions()
        self.statusBar().showMessage(f"已切换项目：{target}；历史对话已按项目筛选")

    def configure_workspace_permissions(self) -> None:
        if not self.ensure_authenticated():
            return
        choices = [
            "只读（每次修改前确认）",
            "自动允许生成和编辑（删除仍确认）",
            "自动允许全部文件操作",
        ]
        selected, accepted = QInputDialog.getItem(
            self,
            "工作区文件操作权限",
            f"当前权限：{workspace_manager.permission_mode}\n请选择权限模式：",
            choices,
            {"read_only": 0, "accept_edits": 1, "accept_all": 2}.get(
                workspace_manager.permission_mode, 0
            ),
            False,
        )
        if not accepted:
            return
        mode_name = ["read_only", "accept_edits", "accept_all"][choices.index(selected)]
        workspace_manager.set_permission_mode(mode_name)
        self.statusBar().showMessage(f"工作区权限已设置：{selected}")

    def _refresh_workspace_label(self) -> None:
        self.workspace_label.setText("当前项目")
        self.workspace_label.setToolTip(f"当前工作区\n{workspace_manager.root}")
        if hasattr(self, "workspace_path_label"):
            self.workspace_path_label.setText(str(workspace_manager.root))
        if hasattr(self, "file_tree"):
            self._refresh_file_tree()
