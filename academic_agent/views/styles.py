"""Academic Agent Qt 客户端的统一样式。"""

from PySide6.QtGui import QColor


BASE_STYLESHEET = """
QMainWindow, QDialog, QWidget { background:#ffffff; color:#2e6f43; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
    background:#ffffff; color:#2e6f43; border:1px solid #bcd9c4;
    border-radius:11px; padding:8px 12px; min-height:24px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus { border:2px solid #43a047; }
QComboBox::drop-down { border:0; width:28px; }
QComboBox QAbstractItemView { background:#ffffff; color:#2e6f43; border:1px solid #bcd9c4; border-radius:10px; padding:6px; outline:0; selection-background-color:#e8f5e9; selection-color:#176b38; }
QFrame#sidebar { background:#ffffff; border-right:1px solid #d7e8dc; }
QFrame#right_sidebar { background:#ffffff; border-left:1px solid #d7e8dc; }
QLabel#brand { color:#176b38; font-size:18px; font-weight:700; padding:14px 10px; }
QLabel#section_title { color:#4f805f; font-size:13px; font-weight:700; padding:12px 8px 5px; }
QListWidget { background:#ffffff; color:#2e6f43; border:0; padding:4px; }
QListWidget::item { padding:9px 11px; margin:1px 2px; border-radius:10px; }
QListWidget::item:selected { background:#e8f5e9; color:#176b38; }
QFrame#topbar { background:#ffffff; border-bottom:1px solid #d7e8dc; }
QLabel#conversation_title { color:#176b38; font-size:15px; font-weight:700; padding:12px 16px; }
QTextBrowser#chat { background:#ffffff; border:0; padding:30px 9%; font-size:15px; }
QWidget#composer_shell { background:#ffffff; }
QFrame#composer_box { background:#ffffff; border:1px solid #bcd9c4; border-radius:18px; }
QFrame#composer_box:focus-within { border:2px solid #43a047; }
QFrame#agent_action_panel { background:#f7fbf8; border:1px solid #bcd9c4; border-radius:15px; margin:0 52px; }
QLabel#agent_action_title { color:#176b38; font-size:15px; font-weight:700; }
QLabel#agent_action_context { color:#2e6f43; font-size:14px; }
QLabel#agent_action_hint { color:#668270; font-size:12px; }
QPlainTextEdit#agent_action_details, QPlainTextEdit#agent_action_input { background:#ffffff; color:#2e6f43; border:1px solid #d7e8dc; border-radius:10px; padding:7px 9px; }
QPlainTextEdit#agent_action_input:focus { border:2px solid #43a047; }
QPushButton#agent_action_submit { background:#4caf50; color:#ffffff; border-color:#388e3c; font-weight:600; }
QPushButton#agent_action_submit:hover { background:#388e3c; }
QPushButton#agent_action_cancel { color:#668270; }
QPushButton#agent_action_option { background:#ffffff; color:#2e6f43; border:1px solid #cfe3d4; border-radius:10px; padding:7px 10px; text-align:left; }
QPushButton#agent_action_option:hover { background:#f1f8f3; border-color:#79b88a; }
QPushButton#agent_action_option:checked { background:#e8f5e9; color:#176b38; border:2px solid #43a047; }
QPlainTextEdit#composer { background:transparent; border:0; padding:12px 14px 4px; font-size:14px; }
QPushButton, QToolButton { border:1px solid #bcd9c4; border-radius:11px; padding:8px 14px; background:#ffffff; color:#2e6f43; min-height:22px; }
QPushButton:hover { background:#f1f8f3; border-color:#79b88a; }
QPushButton#send { background:#4caf50; color:#ffffff; border:1px solid #388e3c; border-radius:17px; padding:8px 20px; font-weight:600; min-width:64px; }
QPushButton#send:hover { background:#388e3c; }
QPushButton#icon_button { min-width:34px; max-width:34px; min-height:34px; max-height:34px; padding:0; border-radius:17px; font-size:18px; }
QPushButton#new_chat { background:#ffffff; border:1px solid #bcd9c4; border-radius:12px; padding:9px 13px; text-align:left; }
QPushButton#new_chat:hover { background:#eff8f1; border-color:#79b88a; }
QPushButton#upload_button { background:#f1f8f3; color:#176b38; border:1px solid #bcd9c4; border-radius:12px; padding:8px 14px; font-weight:600; }
QPushButton#upload_button:hover { background:#e8f5e9; border-color:#79b88a; }
QPushButton#login_button { background:#43a047; color:#ffffff; border:1px solid #388e3c; border-radius:14px; padding:8px 14px; font-weight:600; }
QPushButton#login_button:hover { background:#388e3c; }
QComboBox#feature_select { background:#ffffff; color:#2e6f43; border:1px solid #bcd9c4; border-radius:12px; padding:7px 12px; min-height:27px; }
QComboBox#feature_select:hover { border-color:#79b88a; background:#f8fcf9; }
QComboBox#feature_select::drop-down { border:0; width:24px; }
QComboBox#feature_select QAbstractItemView { background:#ffffff; color:#2e6f43; selection-background-color:#eff8f1; selection-color:#176b38; border:1px solid #bcd9c4; }
QComboBox#mode_selector { background:#ffffff; color:#176b38; border:1px solid #bcd9c4; border-radius:13px; padding:9px 12px; min-height:28px; font-size:14px; font-weight:700; }
QComboBox#mode_selector:hover { border-color:#79b88a; background:#f1f8f3; }
QComboBox#mode_selector::drop-down { border:0; width:28px; }
QComboBox#mode_selector QAbstractItemView { background:#ffffff; color:#2e6f43; selection-background-color:#eff8f1; selection-color:#176b38; border:1px solid #bcd9c4; }
QLabel#approval_mode_label { color:#668270; font-size:12px; padding-left:4px; }
QComboBox#approval_mode_selector { background:#f7fbf8; color:#176b38; border:1px solid #bcd9c4; border-radius:11px; padding:7px 10px; min-width:92px; min-height:24px; font-weight:600; }
QComboBox#approval_mode_selector:hover { border-color:#79b88a; background:#e8f5e9; }
QComboBox#approval_mode_selector QAbstractItemView { background:#ffffff; color:#2e6f43; selection-background-color:#e8f5e9; selection-color:#176b38; border:1px solid #bcd9c4; }
QToolButton#common_toggle { background:#f3faf5; color:#176b38; border:1px solid #c4e1cc; border-radius:12px; text-align:left; padding:9px 12px; font-size:14px; font-weight:700; }
QToolButton#common_toggle:hover { background:#c8e6c9; border-color:#66bb6a; }
QToolButton#workspace_button { background:#ffffff; color:#176b38; border:1px solid #bcd9c4; border-radius:11px; padding:8px 12px; font-weight:600; }
QToolButton#workspace_button:hover { background:#f1f8f3; border-color:#79b88a; }
QToolButton#workspace_button::menu-indicator, QToolButton#account_button::menu-indicator { image:none; width:0; }
QToolButton#open_file { background:#ffffff; border:0; color:#176b38; padding:7px; border-radius:11px; }
QToolButton#open_file:hover { background:#e8f5e9; border:1px solid #79b88a; }
QFrame#file_panel { background:#ffffff; border-left:1px solid #d7e8dc; }
QTreeWidget#file_tree { background:#ffffff; border:1px solid #d7e8dc; border-radius:14px; padding:6px; color:#2e6f43; outline:0; }
QTreeWidget#file_tree::item { padding:6px 5px; border-radius:8px; }
QTreeWidget#file_tree::item:selected { background:#e3f2fd; color:#1769aa; }
QLineEdit#file_filter { background:#ffffff; border:1px solid #bcd9c4; border-radius:12px; padding:9px 12px; }
QScrollArea#common_scroll { background:transparent; border:0; }
QFrame#account_bar { background:#ffffff; border:0; }
QLabel#account_identity { color:#245c36; font-size:14px; font-weight:600; padding:7px 5px; }
QPushButton#theme_button { background:#ffffff; color:#176b38; border:1px solid #bcd9c4; border-radius:11px; padding:7px 11px; }
QPushButton#model_button { background:#ffffff; color:#176b38; border:1px solid #bcd9c4; border-radius:11px; padding:7px 11px; }
QPushButton#account_button { background:#ffffff; color:#176b38; border:1px solid #bcd9c4; border-radius:11px; padding:7px 11px; }
QToolButton#account_button { background:#ffffff; color:#245c36; border:1px solid #d7e8dc; border-radius:19px; padding:3px 9px; min-width:74px; max-width:110px; min-height:38px; max-height:42px; font-weight:700; }
QToolButton#account_button:hover { background:#eef7f0; border:0; }
QPushButton#settings_button { background:#ffffff; color:#176b38; border:1px solid #bcd9c4; border-radius:11px; padding:7px 11px; }
QScrollBar:vertical { background:transparent; width:9px; margin:5px 2px; border-radius:4px; }
QScrollBar::handle:vertical { background:#b5d8be; min-height:36px; border-radius:4px; }
QScrollBar::handle:vertical:hover { background:#79b88a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QLabel#hint { color:#5a8968; padding:4px; }
QLabel#connection { color:#4f805f; font-size:12px; padding:8px; }
QMenu { background:#ffffff; color:#2e6f43; border:1px solid #cfe3d4; border-radius:12px; padding:7px; }
QMenu::item { padding:8px 26px 8px 12px; margin:2px; border-radius:8px; }
QMenu::item:selected { background:#e8f5e9; color:#176b38; }
QMenu::separator { height:1px; background:#d7e8dc; margin:6px 10px; }
QStatusBar { background:#ffffff; color:#5a8968; border-top:1px solid #e3eee6; padding:3px 10px; font-size:12px; min-height:20px; }
QSplitter::handle { background:#eef6f0; }
QToolTip { background:#ffffff; color:#245c36; border:1px solid #bcd9c4; border-radius:8px; padding:6px 9px; }
"""


def build_stylesheet(base: str, accent: QColor) -> str:
    """根据用户选择的绿色主色生成完整样式。"""
    primary = accent.darker(140).name()
    secondary = accent.darker(112).name()
    light = "#e8f5e9"
    action_blue = "#2196f3"
    action_blue_dark = "#1565c0"
    border = "#bcd9c4"
    divider = "#d7e8dc"
    return base + f"""
        QMainWindow, QWidget {{ background:#ffffff; color:{primary}; }}
        QFrame#sidebar {{ background:#ffffff; border-right:1px solid {divider}; }}
        QFrame#right_sidebar {{ background:#ffffff; border-left:1px solid {divider}; }}
        QLabel#brand {{ color:{primary}; font-size:18px; font-weight:700; padding:14px 10px; }}
        QLabel#section_title {{ color:{secondary}; font-size:13px; font-weight:700; padding:12px 8px 5px; }}
        QLabel#conversation_title {{ color:{primary}; font-size:15px; font-weight:700; padding:12px 16px; }}
        QLabel#connection, QLabel#hint {{ color:{secondary}; font-size:12px; }}
        QListWidget {{ background:#ffffff; color:{primary}; font-size:14px; border:0; }}
        QListWidget::item {{ padding:9px 11px; margin:1px 2px; border-radius:10px; }}
        QListWidget::item:selected {{ background:{light}; color:{primary}; }}
        QComboBox#feature_select {{ background:#ffffff; color:{primary}; border:1px solid {border}; border-radius:12px; padding:7px 12px; font-size:14px; min-height:27px; }}
        QComboBox#feature_select:hover {{ border-color:{accent.name()}; background:#f8fcf9; }}
        QComboBox#feature_select QAbstractItemView {{ color:{primary}; selection-background-color:{light}; selection-color:{primary}; border:1px solid {border}; }}
        QComboBox#mode_selector {{ background:#ffffff; color:{primary}; border:1px solid {border}; border-radius:13px; padding:9px 12px; min-height:28px; font-size:14px; font-weight:700; }}
        QComboBox#mode_selector:hover {{ border-color:{accent.name()}; background:{light}; }}
        QComboBox#mode_selector QAbstractItemView {{ background:#ffffff; color:{primary}; selection-background-color:{light}; selection-color:{primary}; border:1px solid {border}; }}
        QToolButton#common_toggle {{ color:{primary}; background:#f4faf5; border:1px solid #c9e3cf; border-radius:12px; padding:9px 12px; font-size:14px; }}
        QToolButton#common_toggle:hover {{ background:{light}; border-color:{accent.name()}; color:{primary}; }}
        QFrame#account_bar {{ background:#ffffff; border:0; }}
        QLabel#account_identity {{ color:{primary}; font-size:14px; font-weight:600; padding:7px 5px; }}
        QPushButton, QToolButton {{ color:{primary}; background:#ffffff; border:1px solid {border}; border-radius:11px; }}
        QPushButton:hover {{ background:{light}; border-color:{accent.name()}; }}
        QPushButton#new_chat, QPushButton#theme_button, QPushButton#model_button, QPushButton#account_button, QPushButton#settings_button {{ color:{primary}; background:#ffffff; border-color:{border}; border-radius:12px; }}
        QPushButton#upload_button {{ background:#f1f8f3; color:{primary}; border:1px solid {border}; border-radius:12px; padding:8px 14px; font-weight:600; }}
        QPushButton#upload_button:hover {{ background:{light}; border-color:{accent.name()}; }}
        QPushButton#login_button {{ background:{accent.name()}; color:#ffffff; border:1px solid {primary}; border-radius:14px; padding:8px 14px; font-weight:600; }}
        QPushButton#login_button:hover {{ background:{primary}; }}
        QToolButton#account_button {{ color:{primary}; background:#ffffff; border:1px solid {border}; border-radius:19px; padding:3px 9px; min-width:74px; max-width:110px; min-height:38px; max-height:42px; font-weight:700; }}
        QToolButton#account_button:hover {{ background:{light}; border:0; }}
        QToolButton#workspace_button {{ color:{primary}; background:#ffffff; border:1px solid {border}; border-radius:11px; padding:8px 12px; font-weight:600; }}
        QToolButton#workspace_button:hover {{ background:{light}; border-color:{accent.name()}; }}
        QToolButton#workspace_button::menu-indicator, QToolButton#account_button::menu-indicator {{ image:none; width:0; }}
        QToolButton#open_file {{ color:{primary}; background:#ffffff; border:0; border-radius:11px; padding:7px; }}
        QToolButton#open_file:hover {{ background:{light}; border:1px solid {accent.name()}; }}
        QPushButton#send {{ background:#4caf50; color:#ffffff; border:1px solid #388e3c; border-radius:17px; padding:8px 20px; }}
        QPushButton#send:hover {{ background:#388e3c; color:#ffffff; }}
        QPushButton#icon_button {{ min-width:34px; max-width:34px; min-height:34px; max-height:34px; padding:0; border-radius:17px; }}
        QPushButton#open_file {{ background:#ffffff; color:{action_blue_dark}; border-color:#90caf9; }}
        QPushButton#open_file:hover {{ background:#e3f2fd; border-color:{action_blue}; color:{action_blue_dark}; }}
        QFrame#topbar, QFrame#file_panel {{ background:#ffffff; border-color:{divider}; }}
        QTextBrowser#chat, QPlainTextEdit#composer {{ background:#ffffff; color:{primary}; }}
        QFrame#agent_action_panel {{ background:#f7fbf8; border:1px solid {border}; border-radius:15px; margin:0 52px; }}
        QLabel#agent_action_title {{ color:{primary}; font-size:15px; font-weight:700; }}
        QLabel#agent_action_context {{ color:{primary}; font-size:14px; }}
        QLabel#agent_action_hint {{ color:{secondary}; font-size:12px; }}
        QPlainTextEdit#agent_action_details, QPlainTextEdit#agent_action_input {{ background:#ffffff; color:{primary}; border:1px solid {divider}; border-radius:10px; padding:7px 9px; }}
        QPlainTextEdit#agent_action_input:focus {{ border:2px solid {accent.name()}; }}
        QPushButton#agent_action_submit {{ background:#4caf50; color:#ffffff; border-color:#388e3c; font-weight:600; }}
        QPushButton#agent_action_submit:hover {{ background:#388e3c; }}
        QPushButton#agent_action_option {{ background:#ffffff; color:{primary}; border:1px solid #cfe3d4; border-radius:10px; padding:7px 10px; text-align:left; }}
        QPushButton#agent_action_option:hover {{ background:{light}; border-color:{accent.name()}; }}
        QPushButton#agent_action_option:checked {{ background:{light}; color:{primary}; border:2px solid {accent.name()}; }}
        QPushButton#agent_action_cancel {{ color:{secondary}; }}
        QLabel#approval_mode_label {{ color:{secondary}; font-size:12px; padding-left:4px; }}
        QComboBox#approval_mode_selector {{ background:#f7fbf8; color:{primary}; border:1px solid {border}; border-radius:11px; padding:7px 10px; min-width:92px; min-height:24px; font-weight:600; }}
        QComboBox#approval_mode_selector:hover {{ border-color:{accent.name()}; background:{light}; }}
        QComboBox#approval_mode_selector QAbstractItemView {{ background:#ffffff; color:{primary}; selection-background-color:{light}; selection-color:{primary}; border:1px solid {border}; }}
        QLineEdit#file_filter {{ color:{primary}; border-color:{border}; border-radius:12px; }}
        QTreeWidget#file_tree {{ background:#ffffff; color:{primary}; border-radius:14px; }}
        QTreeWidget#file_tree::item:selected {{ background:#e3f2fd; color:{action_blue_dark}; }}
        QStatusBar {{ background:#ffffff; color:{secondary}; border-top:1px solid {divider}; }}
        QMenu {{ background:#ffffff; color:{primary}; border:1px solid {border}; border-radius:12px; padding:7px; }}
        QMenu::item {{ padding:8px 26px 8px 12px; margin:2px; border-radius:8px; }}
        QMenu::item:selected {{ background:{light}; color:{primary}; }}
    """
