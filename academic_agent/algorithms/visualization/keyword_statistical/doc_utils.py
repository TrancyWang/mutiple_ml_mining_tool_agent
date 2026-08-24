import pandas as pd
import ast
import re
import json

# ------------- 工具函数（同上，稳健修复） -------------
def safe_fix_string(s):
    if s is None:
        return ""
    s = str(s).strip()
    for ch in ["\ufeff", "\u200b", "\u200c", "\u200d", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"]:
        s = s.replace(ch, "")
    s = s.replace("\n", "").replace("\r", "")
    return s

def fix_to_dict_auto(s):
    s = safe_fix_string(s)
    # 先尝试 ast.literal_eval
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return dict(parsed)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # 自动修复：单引号->双引号, tuple->list, 补引号
    s2 = s.replace("'", '"')
    s2 = re.sub(r'\((.*?)\)', r'[\1]', s2)
    s2 = re.sub(r'([\[,]\s*)([A-Za-z0-9_\u4e00-\u9fa5]+)\s*(,)', r'\1"\2"\3', s2)
    try:
        data = json.loads(s2)
        if isinstance(data, list):
            return {k: v for k, v in data}
        if isinstance(data, dict):
            return data
    except Exception:
        raise ValueError(f"无法修复为 JSON，可疑内容：{repr(s)}")

# 判断一个字符串是否看起来像 list-of-(key,value) 的候选项
_tuple_like_re = re.compile(r"\(\s*['\"]?.+?['\"]?\s*,\s*[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*\)")

def looks_like_pairs(s):
    if not isinstance(s, str):
        return False
    s = safe_fix_string(s)
    # 如果含有像 ( '词', 1.2 ) 的样式，就认为是候选
    return bool(_tuple_like_re.search(s))