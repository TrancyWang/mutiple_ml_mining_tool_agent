"""PySide6 Academic Agent 客户端兼容入口。

界面实现已经拆分到 ``academic_agent.views`` 包中。本文件保留原有启动路径，避免
``python qt_client.py``、``start_qt.sh`` 和已有外部调用失效。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# macOS Qt 输入法/窗口图层兼容配置，必须在导入 PySide6 前设置。
if sys.platform == "darwin":
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

# Matplotlib/fontconfig 由可视化工具按需导入。显式提供可写缓存目录，避免
# 打包应用每次启动 Agent 都重新扫描系统字体。
cache_root = Path(
    os.getenv(
        "ACADEMIC_AGENT_CACHE_DIR",
        str(Path.home() / "Library" / "Caches" / "AcademicAgent"),
    )
).expanduser()
try:
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
except OSError:
    pass

from PySide6.QtCore import QLockFile, QSettings, QTimer
from academic_agent.infrastructure.model_paths import normalize_model_root

_saved_model_root = str(
    QSettings("AcademicAgent", "Models").value("pretrained_models_dir", "") or ""
).strip()
if _saved_model_root and normalize_model_root(_saved_model_root).is_dir():
    os.environ.setdefault("PRETRAINED_MODELS_DIR", _saved_model_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from academic_agent.infrastructure.storage_config import apply_persisted_storage_config

# 必须在 Agent/记忆服务导入前应用，且在 .env 加载后会在 main() 中再次应用。
apply_persisted_storage_config()

def main() -> None:
    from academic_agent.infrastructure.runtime_paths import output_root

    # 所有图片、表格和文档产物统一写入应用同级 output；启动时即创建。
    output_root()
    app = QApplication(sys.argv)
    icon_path = Path(__file__).resolve().parent / "resources" / "academic_agent_icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    # macOS 双击同一个 .app 可以创建多个进程；如果没有单实例保护，
    # 每个进程都会各自创建登录框和主窗口。锁文件位于系统临时目录，
    # 因此源码运行、打包运行和不同工作区也共用同一个单实例约束。
    lock_path = Path(tempfile.gettempdir()) / "academic_agent_qt_single_instance.lock"
    instance_lock = QLockFile(str(lock_path))
    instance_lock.setStaleLockTime(60_000)
    if not instance_lock.tryLock(100):
        return
    # 保持 Python 引用直到 QApplication 退出，否则锁可能被提前释放。
    app._academic_agent_instance_lock = instance_lock  # type: ignore[attr-defined]

    from academic_agent.controllers.auth import AuthController

    auth_controller = AuthController()

    # 默认以访客状态进入主界面，不自动恢复任何历史账号（包括 trancy）。
    # 用户使用功能时仍可主动登录；登录信息只用于登录框的手动填充。
    session = None
    if session is not None:
        auth_controller.apply(session)
        apply_persisted_storage_config()

    # 认证配置完成后导入 Agent 相关模块；访客状态不会初始化云端 Agent。
    from academic_agent.views.main_window import CodexChatWindow

    window = CodexChatWindow(auth_session=session)
    window.show()
    # 主界面先显示；依赖安装提示在窗口显示后再弹出，避免启动阶段被模态框挡住。
    from academic_agent.infrastructure.dependency_manager import missing_dependencies
    from academic_agent.views.dialogs import DependencyDialog

    missing = missing_dependencies()
    if missing:
        QTimer.singleShot(0, lambda: DependencyDialog(missing, window).exec())
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


def __getattr__(name):
    """兼容旧代码从 qt_client 导入界面类，同时保持启动时延迟导入。"""
    if name == "CodexChatWindow":
        from academic_agent.views.main_window import CodexChatWindow

        return CodexChatWindow
    if name == "DependencyDialog":
        from academic_agent.views.dialogs import DependencyDialog

        return DependencyDialog
    if name == "StreamWorker":
        from academic_agent.views.workers import StreamWorker

        return StreamWorker
    raise AttributeError(name)


__all__ = ["CodexChatWindow", "DependencyDialog", "StreamWorker", "main"]
