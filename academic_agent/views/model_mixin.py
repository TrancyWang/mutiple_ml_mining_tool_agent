"""Qt 客户端的文本挖掘模型库设置。"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import (
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
        title: str = "设置模型库根目录",
        description: str = (
            "请选择包含 bge-cn、multilingual-sentiment-analysis 等子目录的 "
            "pretrain_models 文件夹："
        ),
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
                hint.setText("未指定：将使用默认模型库目录。")
                return
            status = model_components(root)
            components = (
                ("BGE 向量模型", status["has_bge"]),
                ("通用五分类模型", status["has_general_sentiment"]),
                ("中文八分类情绪模型", status["has_chinese_sentiment"]),
            )
            component_lines = "\n".join(
                f"{'✓' if found else '○'} {name}：{'已找到' if found else '未找到'}"
                for name, found in components
            )
            hint.setText(f"模型库根目录：{root}\n{component_lines}")

        path_edit.textChanged.connect(update_hint)
        update_hint()
        dialog._path_edit = path_edit  # type: ignore[attr-defined]
        return dialog

    def _choose_model_directory(self, path_edit: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择模型库根目录（pretrain_models）",
            path_edit.text().strip() or str(Path.home()),
        )
        if selected:
            path_edit.setText(selected)

    def configure_model_directory(
        self,
        title: str = "设置模型库根目录",
        description: str = (
            "请选择包含 bge-cn、multilingual-sentiment-analysis 等子目录的 "
            "pretrain_models 文件夹："
        ),
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
                "模型库目录不完整",
                f"目录中没有找到 BGE 向量模型：\n{root / 'bge-cn'}\n\n"
                "请选择包含 bge-cn 子目录的 pretrain_models 文件夹。",
            )
            return False
        apply_model_root(root)
        self.model_settings.setValue("pretrained_models_dir", str(root))
        from academic_agent.infrastructure.persistence.chroma_store import vector_store
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter

        vector_store.set_model_root(root)
        source_video_text_adapter.set_model_root(root)
        self.statusBar().showMessage(f"模型库根目录已设置：{root}")
        return True

    def configure_text_mining_model(self) -> bool:
        """在执行文本挖掘前，让用户补充算法模型目录。"""
        return self.configure_model_directory(
            title="设置文本挖掘模型库",
            description=(
                "文本挖掘算法需要模型库。请选择包含 bge-cn 子目录的 "
                "pretrain_models 文件夹；通用情感模型也应放在此目录内。"
            ),
        )

    def configure_sentiment_model(self) -> bool:
        if not self.ensure_authenticated():
            return False
        current = str(self.model_settings.value("sentiment_model_path", "") or "")
        path = QFileDialog.getExistingDirectory(
            self,
            "选择中文八分类情绪模型目录",
            current or str(Path.home()),
        )
        if not path:
            return False
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_dir():
            QMessageBox.warning(self, "模型目录无效", f"目录不存在：\n{model_path}")
            return False
        if not (model_path / "config.json").is_file():
            QMessageBox.warning(
                self,
                "模型目录无效",
                "这里应选择具体的中文八分类模型文件夹，并且目录中应包含 config.json。\n\n"
                "例如：pretrain_models/xuyuan-trial-sentiment-bert-chinese",
            )
            return False
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter
        source_video_text_adapter.set_sentiment_model_path(model_path)
        self.model_settings.setValue("sentiment_model_path", str(model_path))
        self.statusBar().showMessage(f"中文八分类情绪模型已设置：{model_path}")
        return True

    def reset_sentiment_model(self) -> None:
        if not self.ensure_authenticated():
            return
        from academic_agent.integrations.video_text_adapter import source_video_text_adapter
        source_video_text_adapter.reset_sentiment_model_path()
        self.model_settings.remove("sentiment_model_path")
        self.statusBar().showMessage("已清除单独指定的中文八分类模型，将使用模型库中的默认模型")

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
        self.statusBar().showMessage("已恢复默认模型库根目录")
