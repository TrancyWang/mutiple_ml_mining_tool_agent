"""Qt 客户端的本地模型目录设置。"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from academic_agent.infrastructure.model_paths import apply_model_root, model_components, normalize_model_root


class ModelMixin:
    """让用户手动填写或选择本地模型目录。"""

    def _build_model_path_dialog(
        self,
        title: str = "设置本地模型目录",
        description: str = "填写模型根目录，或选择包含 bge-cn 子目录的文件夹：",
    ) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(620)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(description))

        row = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setPlaceholderText("例如：/Volumes/models/pretrain_models")
        current = os.getenv("PRETRAINED_MODELS_DIR", "")
        path_edit.setText(current)
        row.addWidget(path_edit, 1)
        browse_btn = QPushButton("选择目录")
        browse_btn.clicked.connect(lambda: self._choose_model_directory(path_edit))
        row.addWidget(browse_btn)
        layout.addLayout(row)

        hint = QLabel()
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def update_hint() -> None:
            root = normalize_model_root(path_edit.text())
            if root is None:
                hint.setText("未指定：将使用默认模型目录。")
                return
            status = model_components(root)
            missing = []
            if not status["has_bge"]:
                missing.append("bge-cn")
            if not status["has_sentiment"]:
                missing.append("xuyuan-trial-sentiment-bert-chinese（可选）")
            suffix = "；缺少：" + "、".join(missing) if missing else "；BGE 和情感模型均已找到"
            hint.setText(f"模型根目录：{root}{suffix}")

        path_edit.textChanged.connect(update_hint)
        update_hint()
        dialog._path_edit = path_edit  # type: ignore[attr-defined]
        return dialog

    def _choose_model_directory(self, path_edit: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择本地模型根目录",
            path_edit.text().strip() or str(Path.home()),
        )
        if selected:
            path_edit.setText(selected)

    def configure_model_directory(
        self,
        title: str = "设置本地模型目录",
        description: str = "填写模型根目录，或选择包含 bge-cn 子目录的文件夹：",
    ) -> bool:
        if not self.ensure_authenticated():
            return False
        dialog = self._build_model_path_dialog(title, description)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        root = normalize_model_root(dialog._path_edit.text())  # type: ignore[attr-defined]
        if root is None:
            self.reset_model_directory()
            return True
        status = model_components(root)
        if not status["has_bge"]:
            QMessageBox.warning(
                self,
                "模型目录无效",
                f"目录中没有找到 bge-cn：\n{root / 'bge-cn'}\n\n请重新选择模型根目录。",
            )
            return False
        apply_model_root(root)
        self.model_settings.setValue("pretrained_models_dir", str(root))
        from academic_agent.infrastructure.persistence.chroma_store import vector_store
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter

        vector_store.set_model_root(root)
        source_video_text_adapter.set_model_root(root)
        self.statusBar().showMessage(f"本地模型目录已设置：{root}")
        return True

    def configure_text_mining_model(self) -> bool:
        """在执行文本挖掘前，让用户补充算法模型目录。"""
        return self.configure_model_directory(
            title="设置文本挖掘模型地址",
            description=(
                "文本挖掘算法需要本地模型目录。请填写或选择包含 "
                "bge-cn 子目录的文件夹："
            ),
        )

    def configure_sentiment_model(self) -> bool:
        if not self.ensure_authenticated():
            return False
        current = str(self.model_settings.value("sentiment_model_path", "") or "")
        path = QFileDialog.getExistingDirectory(
            self, "选择情感分析模型目录", current or str(Path.home())
        )
        if not path:
            return False
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_dir():
            QMessageBox.warning(self, "模型目录无效", f"目录不存在：\n{model_path}")
            return False
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter
        source_video_text_adapter.set_sentiment_model_path(model_path)
        self.model_settings.setValue("sentiment_model_path", str(model_path))
        self.statusBar().showMessage(f"情感分析模型地址已设置：{model_path}")
        return True

    def reset_sentiment_model(self) -> None:
        if not self.ensure_authenticated():
            return
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter
        source_video_text_adapter.reset_sentiment_model_path()
        self.model_settings.remove("sentiment_model_path")
        self.statusBar().showMessage("已恢复默认情感分析模型")

    def configure_custom_dictionary(self) -> bool:
        if not self.ensure_authenticated():
            return False
        current = str(self.model_settings.value("custom_dictionary_path", "") or "")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择自定义词典", current or str(Path.home()),
            "词典文件 (*.txt *.dict *.dic);;所有文件 (*)"
        )
        if not path:
            return False
        dictionary = Path(path).expanduser().resolve()
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter
        source_video_text_adapter.set_custom_dictionary(dictionary)
        self.model_settings.setValue("custom_dictionary_path", str(dictionary))
        self.statusBar().showMessage(f"自定义词典已设置：{dictionary.name}")
        return True

    def reset_custom_dictionary(self) -> None:
        if not self.ensure_authenticated():
            return
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter
        source_video_text_adapter.reset_custom_dictionary()
        self.model_settings.remove("custom_dictionary_path")
        self.statusBar().showMessage("已停用自定义词典")

    def reset_model_directory(self) -> None:
        if not self.ensure_authenticated():
            return
        apply_model_root(None)
        self.model_settings.remove("pretrained_models_dir")
        from academic_agent.infrastructure.persistence.chroma_store import vector_store
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter

        vector_store.reset_model_root()
        source_video_text_adapter.reset_model_root()
        self.statusBar().showMessage("已恢复默认模型目录")
