"""Academic Agent Qt 客户端的统一样式。"""

from PySide6.QtGui import QColor


DEFAULT_ACCENT = "#8fd3a1"

THEME_SAGE = "sage"
THEME_SKY = "sky"
THEME_LAVENDER = "lavender"
THEME_SAND = "sand"
# 旧名称保留为兼容别处的导入；现在所有可选主题都是浅色方案。
THEME_LIGHT = THEME_SAGE
THEME_OPTIONS = (
    ("清新薄荷绿（默认）", THEME_SAGE),
    ("清透天空蓝", THEME_SKY),
    ("淡雅薰衣草", THEME_LAVENDER),
    ("晨光奶油黄", THEME_SAND),
)

THEME_PRESETS = {
    THEME_SAGE: {
        "accent": DEFAULT_ACCENT,
        "primary": "#52745b",
        "secondary": "#789783",
        "light": "#eef9f1",
        "canvas": "#eef3ef",
        "surface": "#fbfdfb",
        "sidebar": "#f7faf7",
        "input": "#ffffff",
        "border": "#d1e7d6",
        "divider": "#e0eee3",
        "selection": "#dff1e3",
    },
    THEME_SKY: {
        "accent": "#8fc9e8",
        "primary": "#3e6680",
        "secondary": "#7291a5",
        "light": "#eef7fc",
        "canvas": "#edf4f8",
        "surface": "#fbfdff",
        "sidebar": "#f5faff",
        "input": "#ffffff",
        "border": "#c8dfed",
        "divider": "#dfeaf2",
        "selection": "#dceef8",
    },
    THEME_LAVENDER: {
        "accent": "#c4b4e6",
        "primary": "#685f84",
        "secondary": "#8c84a7",
        "light": "#f5f1fb",
        "canvas": "#f1eff7",
        "surface": "#fdfbff",
        "sidebar": "#f8f5fc",
        "input": "#ffffff",
        "border": "#ded6ef",
        "divider": "#ebe6f4",
        "selection": "#ece5f8",
    },
    THEME_SAND: {
        "accent": "#f1d58d",
        "primary": "#7a7046",
        "secondary": "#a59b70",
        "light": "#fffbeb",
        "canvas": "#f7f5ea",
        "surface": "#fffefa",
        "sidebar": "#fcfbf1",
        "input": "#ffffff",
        "border": "#eee4b9",
        "divider": "#f1edd7",
        "selection": "#fff2c5",
    },
}


def normalize_theme_mode(value: object) -> str:
    """将设置值收敛到浅色主题，兼容旧版本的 dark/auto 设置。"""
    mode = str(value or THEME_LIGHT).strip().lower()
    return mode if mode in THEME_PRESETS else THEME_LIGHT


def theme_preset(mode: str) -> dict[str, str]:
    """返回当前浅色皮肤的颜色变量。"""
    return THEME_PRESETS[normalize_theme_mode(mode)]


def theme_chat_colors(accent: QColor, mode: str) -> dict[str, str]:
    """返回 QTextBrowser 内联 HTML 使用的主题色。"""
    palette = theme_preset(mode)
    return {
        "text": palette["primary"],
        "strong": accent.darker(145).name(),
        "muted": palette["secondary"],
        "surface": palette["input"],
        "surface_alt": palette["light"],
        "border": palette["border"],
        "accent": accent.name(),
        "accent_text": "#ffffff",
        "link": "#1769aa",
        "success": palette["primary"],
        "warning": "#b26a00",
        "error": "#a33f3f",
    }


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
QPushButton#voice_button { background:#fff8e1; color:#7a5a00; border:1px solid #e5c76b; border-radius:12px; padding:8px 14px; font-weight:600; }
QPushButton#voice_button:hover { background:#fff3c4; border-color:#c9a227; }
QPushButton#voice_button:disabled { background:#f6f6f6; color:#8a8a8a; border-color:#d6d6d6; }
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


def build_stylesheet(base: str, accent: QColor, theme_mode: str = THEME_LIGHT) -> str:
    """根据主题模式和主色生成完整样式。"""
    theme_mode = normalize_theme_mode(theme_mode)
    preset = theme_preset(theme_mode)
    primary = preset["primary"]
    secondary = preset["secondary"]
    light = preset["light"]
    action_blue = "#2196f3"
    action_blue_dark = "#1565c0"
    border = preset["border"]
    divider = preset["divider"]
    accent_border = preset["border"]
    accent_hover = preset["light"]
    stylesheet = base + f"""
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
        QPushButton#voice_button {{ background:#fff8e1; color:#7a5a00; border:1px solid #e5c76b; border-radius:12px; padding:8px 14px; font-weight:600; }}
        QPushButton#voice_button:hover {{ background:#fff3c4; border-color:#c9a227; }}
        QPushButton#voice_button:disabled {{ background:#f6f6f6; color:#8a8a8a; border-color:#d6d6d6; }}
        QPushButton#login_button {{ background:{accent.name()}; color:#ffffff; border:1px solid {primary}; border-radius:14px; padding:8px 14px; font-weight:600; }}
        QPushButton#login_button:hover {{ background:{primary}; }}
        QToolButton#account_button {{ color:{primary}; background:#ffffff; border:1px solid {border}; border-radius:19px; padding:3px 9px; min-width:74px; max-width:110px; min-height:38px; max-height:42px; font-weight:700; }}
        QToolButton#account_button:hover {{ background:{light}; border:0; }}
        QToolButton#workspace_button {{ color:{primary}; background:#ffffff; border:1px solid {border}; border-radius:11px; padding:8px 12px; font-weight:600; }}
        QToolButton#workspace_button:hover {{ background:{light}; border-color:{accent.name()}; }}
        QToolButton#workspace_button::menu-indicator, QToolButton#account_button::menu-indicator {{ image:none; width:0; }}
        QToolButton#open_file {{ color:{primary}; background:#ffffff; border:0; border-radius:11px; padding:7px; }}
        QToolButton#open_file:hover {{ background:{light}; border:1px solid {accent.name()}; }}
        QPushButton#send {{ background:{accent.name()}; color:{primary}; border:1px solid {accent_border}; border-radius:17px; padding:8px 20px; }}
        QPushButton#send:hover {{ background:{accent_hover}; color:{primary}; }}
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
        QPushButton#agent_action_submit {{ background:{accent.name()}; color:{primary}; border-color:{accent_border}; font-weight:600; }}
        QPushButton#agent_action_submit:hover {{ background:{accent_hover}; }}
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

        /* Product shell: quiet canvas, clear panel hierarchy, one primary action. */
        QWidget#app_root {{ background:#eef3ef; }}
        QFrame#sidebar, QFrame#main_panel, QFrame#right_sidebar {{
            background:#fbfdfb; border:1px solid #dce8df; border-radius:16px;
        }}
        QFrame#sidebar {{ background:#f7faf7; }}
        QFrame#right_sidebar {{ background:#f6faf7; }}
        QFrame#brand_bar {{ background:transparent; border:0; }}
        QLabel#brand_mark {{
            background:#dff2e3; color:{primary}; border:1px solid {accent_border}; border-radius:10px;
            font-size:18px; font-weight:800;
        }}
        QLabel#brand_name {{ color:{primary}; font-size:15px; font-weight:800; }}
        QLabel#brand_subtitle {{ color:{secondary}; font-size:11px; }}
        QComboBox#mode_selector {{
            background:#eaf5ed; color:{primary}; border:1px solid #cfe3d4;
            border-radius:10px; padding:8px 11px; min-height:27px; font-size:13px; font-weight:700;
        }}
        QComboBox#mode_selector:hover {{ background:{light}; border-color:{accent.name()}; }}
        QPushButton#new_chat {{
            background:#d9f0de; color:{primary}; border:1px solid {accent_border}; border-radius:10px;
            padding:9px 13px; min-height:28px; font-weight:700; text-align:left;
        }}
        QPushButton#new_chat:hover {{ background:{accent_hover}; border-color:{accent.name()}; }}
        QLabel#section_title {{ color:{secondary}; font-size:11px; letter-spacing:1px; padding:10px 4px 4px; }}
        QListWidget {{ background:transparent; color:{primary}; border:0; padding:2px 0; }}
        QListWidget::item {{ padding:8px 10px; margin:1px 0; border-radius:9px; }}
        QListWidget::item:hover {{ background:#edf6ef; }}
        QListWidget::item:selected {{ background:#dff1e3; color:{primary}; font-weight:600; }}
        QToolButton#workspace_button {{ background:#ffffff; border:1px solid #d5e4d9; border-radius:9px; padding:7px 9px; font-size:12px; }}
        QToolButton#workspace_button:hover {{ background:{light}; border-color:{accent.name()}; }}
        QLabel#connection {{ color:{secondary}; font-size:11px; padding:5px 3px; }}
        QFrame#account_bar {{ background:transparent; border-top:1px solid #e1ebe3; }}
        QLabel#account_identity {{ color:{primary}; font-size:12px; font-weight:700; padding:7px 3px; }}
        QPushButton#login_button {{ background:#d9f0de; color:{primary}; border:1px solid {accent_border}; border-radius:10px; padding:8px 12px; }}
        QToolButton#account_button {{ background:#ffffff; border:1px solid #d5e4d9; border-radius:18px; min-height:34px; }}
        QFrame#topbar {{ background:#ffffff; border:0; border-bottom:1px solid #e3ebe5; border-top-left-radius:16px; border-top-right-radius:16px; }}
        QLabel#conversation_title {{ color:{primary}; font-size:16px; font-weight:800; padding:0; }}
        QLabel#topbar_subtitle {{ color:{secondary}; font-size:11px; padding:0; }}
        QLabel#topbar_badge {{ background:#eaf5ed; color:{primary}; border:1px solid #cfe3d4; border-radius:9px; padding:6px 10px; font-size:11px; font-weight:700; }}
        QToolButton#open_file {{ color:{primary}; background:#f5faf6; border:1px solid #d5e4d9; border-radius:9px; padding:6px; }}
        QToolButton#open_file:hover {{ background:{light}; border-color:{accent.name()}; }}
        QTextBrowser#chat {{ background:#ffffff; color:{primary}; padding:20px 7%; font-size:14px; }}
        QFrame#composer_box {{ background:#ffffff; border:1px solid #cfe0d4; border-radius:18px; }}
        QFrame#composer_box:focus-within {{ border:2px solid {accent.name()}; }}
        QPlainTextEdit#composer {{ background:transparent; color:{primary}; border:0; padding:11px 12px 2px; font-size:14px; }}
        QLabel#composer_hint {{ color:#8a9b8e; font-size:11px; }}
        QPushButton#upload_button, QPushButton#voice_button, QPushButton#model_button {{
            background:#f5f9f6; color:{primary}; border:1px solid #d5e4d9; border-radius:9px;
            padding:7px 10px; min-height:22px; font-size:12px; font-weight:600;
        }}
        QPushButton#upload_button:hover, QPushButton#voice_button:hover, QPushButton#model_button:hover {{ background:{light}; border-color:{accent.name()}; }}
        QPushButton#send {{ background:{accent.name()}; color:{primary}; border:1px solid {accent_border}; border-radius:10px; padding:8px 19px; min-width:68px; min-height:24px; font-weight:800; }}
        QPushButton#send:hover {{ background:{accent_hover}; border-color:{accent.name()}; }}
        QLabel#approval_mode_label {{ color:{secondary}; font-size:11px; padding-left:4px; }}
        QComboBox#approval_mode_selector {{ background:#f5f9f6; color:{primary}; border:1px solid #d5e4d9; border-radius:9px; padding:7px 9px; min-width:82px; min-height:22px; font-size:12px; font-weight:600; }}
        QFrame#panel_header {{ background:transparent; border:0; }}
        QLabel#panel_title {{ color:{primary}; font-size:15px; font-weight:800; }}
        QLabel#panel_subtitle {{ color:{secondary}; font-size:11px; }}
        QFrame#feature_group {{ background:#f8fcf9; border:1px solid #dcece0; border-radius:12px; }}
        QLabel#feature_group_title {{ color:{primary}; font-size:13px; font-weight:800; padding:1px 0; }}
        QLabel#feature_group_hint {{ color:{secondary}; font-size:11px; padding-bottom:2px; }}
        QToolButton#common_toggle {{ background:#ffffff; color:{primary}; border:1px solid #d5e4d9; border-radius:10px; padding:8px 10px; font-size:12px; font-weight:700; }}
        QToolButton#common_toggle:hover {{ background:{light}; border-color:{accent.name()}; }}
        QComboBox#feature_select {{ background:#ffffff; color:{primary}; border:1px solid #d5e4d9; border-radius:9px; padding:7px 10px; min-height:24px; font-size:12px; }}
        QComboBox#feature_select:hover {{ background:#f1f8f3; border-color:{accent.name()}; }}
        QFrame#agent_action_panel {{ background:#ffffff; border:1px solid #cfe0d4; border-left:4px solid {accent.name()}; border-radius:12px; margin:0 7%; }}
        QStatusBar {{ background:#f7faf7; color:{secondary}; border-top:1px solid #dce8df; padding:3px 10px; font-size:11px; min-height:19px; }}
        QSplitter::handle {{ background:transparent; }}
    """
    # 末尾再覆盖一遍产品壳层里仍然写死的绿色值，使蓝、紫、橙三套浅色
    # 皮肤真正有明显差异，同时不改变控件层级和布局。
    if theme_mode == THEME_SAGE:
        return stylesheet
    return stylesheet + f"""
        QMainWindow, QDialog, QWidget {{ background:{preset["canvas"]}; color:{primary}; }}
        QWidget#app_root {{ background:{preset["canvas"]}; }}
        QFrame#sidebar, QFrame#main_panel, QFrame#right_sidebar {{
            background:{preset["surface"]}; border-color:{preset["divider"]};
        }}
        QFrame#sidebar {{ background:{preset["sidebar"]}; }}
        QFrame#right_sidebar {{ background:{preset["sidebar"]}; }}
        QFrame#topbar, QFrame#file_panel {{ background:{preset["surface"]}; border-color:{preset["divider"]}; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit,
        QPlainTextEdit, QTreeWidget, QTextBrowser {{
            background:{preset["input"]}; color:{primary}; border-color:{border};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
        QDateEdit:focus, QPlainTextEdit:focus {{ border-color:{accent.name()}; }}
        QComboBox QAbstractItemView, QComboBox#feature_select QAbstractItemView,
        QComboBox#mode_selector QAbstractItemView,
        QComboBox#approval_mode_selector QAbstractItemView {{
            background:{preset["input"]}; color:{primary}; border-color:{border};
            selection-background-color:{preset["selection"]}; selection-color:{primary};
        }}
        QComboBox#mode_selector, QComboBox#feature_select,
        QComboBox#approval_mode_selector {{ background:{preset["light"]}; color:{primary}; border-color:{border}; }}
        QListWidget {{ background:transparent; color:{primary}; }}
        QListWidget::item:hover {{ background:{preset["light"]}; }}
        QListWidget::item:selected {{ background:{preset["selection"]}; color:{primary}; }}
        QPushButton, QToolButton {{ background:{preset["input"]}; color:{primary}; border-color:{border}; }}
        QPushButton:hover, QToolButton:hover {{ background:{preset["light"]}; border-color:{accent.name()}; }}
        QPushButton#new_chat, QPushButton#login_button {{ background:{preset["selection"]}; color:{primary}; border-color:{border}; }}
        QPushButton#send, QPushButton#agent_action_submit {{ background:{accent.name()}; color:{primary}; border-color:{border}; }}
        QPushButton#upload_button, QPushButton#voice_button, QPushButton#model_button,
        QToolButton#workspace_button, QToolButton#account_button,
        QToolButton#open_file, QToolButton#common_toggle,
        QPushButton#theme_button, QPushButton#settings_button {{
            background:{preset["input"]}; color:{primary}; border-color:{border};
        }}
        QTextBrowser#chat {{ background:{preset["input"]}; color:{primary}; }}
        QFrame#composer_box, QFrame#agent_action_panel, QFrame#feature_group,
        QFrame#settings_card {{ background:{preset["surface"]}; border-color:{border}; }}
        QPlainTextEdit#agent_action_details, QPlainTextEdit#agent_action_input {{
            background:{preset["input"]}; color:{primary}; border-color:{border};
        }}
        QTreeWidget#file_tree {{ background:{preset["input"]}; color:{primary}; border-color:{border}; }}
        QMenu {{ background:{preset["input"]}; color:{primary}; border-color:{border}; }}
        QMenu::item:selected {{ background:{preset["selection"]}; color:{primary}; }}
        QStatusBar {{ background:{preset["sidebar"]}; color:{secondary}; border-color:{preset["divider"]}; }}
        QToolTip {{ background:{preset["input"]}; color:{primary}; border-color:{border}; }}
    """
