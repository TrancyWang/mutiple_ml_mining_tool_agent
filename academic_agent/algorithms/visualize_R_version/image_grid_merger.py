# -*- coding: utf-8 -*-
"""
图片网格整合工具
将同一批（如文件夹模式下多个文件生成的）图片拼接到同一张网格图中。
支持自动计算行列数、统一尺寸、添加标题标签，并保留原图。
"""

import os
from PIL import Image, ImageDraw, ImageFont


def _load_font(size):
    """尝试加载一个合适大小的字体，失败则返回默认字体"""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _is_ascii(text):
    """判断字符串是否仅含 ASCII 字符（避免合并图上出现中文）"""
    return all(ord(ch) < 128 for ch in str(text))


def _wrap_text(text, font, max_width, draw):
    """按像素宽度对文本做简单换行；中文按字符换行，英文按词换行。"""
    text = str(text)
    # 先尝试按原文测量，若整体不超宽则直接返回
    if draw.textlength(text, font=font) <= max_width:
        return [text]

    # 中文/全角字符逐个断，英文按空格断
    lines = []
    cur = ""
    for ch in text:
        test = cur + ch
        if draw.textlength(test, font=font) > max_width:
            # 当前行已满，尝试在上一个空格处断开
            if ch == " " and " " in cur:
                lines.append(cur.rstrip())
                cur = ""
            elif ord(ch) > 127:
                # 中文：直接在此字符前断开
                if cur:
                    lines.append(cur)
                    cur = ch
                else:
                    cur = ch
            else:
                # 英文长词：尽量按词切
                if " " in cur:
                    pre, _, post = cur.rpartition(" ")
                    lines.append(pre)
                    cur = post + ch
                else:
                    # 无空格长串（如长文件名）：逐字符断行，避免单行超宽被裁切
                    if len(cur) > 1:
                        lines.append(cur[:-1])
                        cur = cur[-1] + ch
                    else:
                        lines.append(cur)
                        cur = ch
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines if lines else [text]


def _fit_grid(n, prefer_cols=None):
    """根据图片数量计算 (cols, rows) 网格布局"""
    if prefer_cols and prefer_cols > 0:
        cols = min(prefer_cols, n)
    else:
        # 接近正方形的布局
        cols = int(n ** 0.5)
        if cols < 1:
            cols = 1
        while cols * cols < n:
            cols += 1
        # 不要超过 4 列，避免过扁
        cols = min(cols, 4)
    rows = (n + cols - 1) // cols
    return cols, rows


def merge_images_to_grid(
    image_paths,
    output_path,
    title=None,
    labels=None,
    prefer_cols=None,
    cell_size=None,
    bg_color=(255, 255, 255),
):
    """
    将多张图片拼接为网格图。

    Parameters
    ----------
    image_paths : list[str]
        待拼接的图片路径列表（按期望顺序）
    output_path : str
        输出拼接图的路径
    title : str, optional
        顶部总标题
    labels : list[str], optional
        每个格子下方的小标题（与 image_paths 顺序对应）
    prefer_cols : int, optional
        期望列数，默认自动计算（不超过 4）
    cell_size : tuple[int,int], optional
        每个格子统一缩放到的尺寸，默认使用图片原始尺寸的最大值
    bg_color : tuple
        背景色，默认白色

    Returns
    -------
    str
        生成的拼接图路径
    """
    valid = []
    valid_labels = []
    for i, p in enumerate(image_paths):
        if p and os.path.isfile(p):
            valid.append(p)
            if labels and i < len(labels):
                raw = labels[i]
            else:
                raw = os.path.splitext(os.path.basename(p))[0]
            # 中文（非 ASCII）不出现在合并图上 → 用序号代替；
            # 其余名字完整保留，靠自动换行显示完全，不做截断
            valid_labels.append(raw if _is_ascii(raw) else f"Image {len(valid) + 1}")

    if not valid:
        raise ValueError("没有有效的图片可供拼接")

    if len(valid) == 1:
        # 单张直接复制（保持原图，便于统一处理）
        from shutil import copyfile
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        copyfile(valid[0], output_path)
        return output_path

    imgs = [Image.open(p).convert("RGB") for p in valid]

    # 决定每个格子的统一尺寸
    if cell_size:
        tw, th = cell_size
    else:
        max_w = max(im.width for im in imgs)
        max_h = max(im.height for im in imgs)
        tw, th = max_w, max_h

    cols, rows = _fit_grid(len(valid), prefer_cols)
    # 字体调小，避免说明文字显示不全
    font_size = max(10, int(min(tw, th) * 0.032))
    font = _load_font(font_size)
    line_height = int(font_size * 1.3)

    # 边距
    pad = int(min(tw, th) * 0.02) + 4
    grid_w = cols * tw + (cols + 1) * pad
    title_height = 0
    title_font = None
    title_lines = []
    if title:
        # 标题含非 ASCII（如中文）则跳过，避免合并图出现中文
        title = title if _is_ascii(title) else None
    if title:
        title_font = _load_font(max(11, int(font_size * 1.5)))
        _d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        title_lines = _wrap_text(title, title_font, grid_w - 2 * pad, _d) \
            if grid_w - 2 * pad > 0 else [title]
        title_height = int(len(title_lines) * font_size * 1.7 + pad * 0.5)

    # 计算每个标签的换行行数，动态决定 label 区高度
    label_lines_list = []
    max_lines = 0
    if valid_labels:
        canvas_tmp = Image.new("RGB", (1, 1))
        d_tmp = ImageDraw.Draw(canvas_tmp)
        for lbl in valid_labels:
            lines = _wrap_text(lbl, font, tw - 8, d_tmp)
            label_lines_list.append(lines)
            max_lines = max(max_lines, len(lines))
    label_height = max_lines * line_height + int(line_height * 0.3) if valid_labels else 0

    grid_h = rows * (th + label_height) + (rows + 1) * pad + title_height

    canvas = Image.new("RGB", (grid_w, grid_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    if title and title_font:
        ty = int(pad * 0.5)
        for line in title_lines:
            draw.text((pad, ty), line, fill=(0, 0, 0), font=title_font)
            ty += int(font_size * 1.7)

    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tw + pad)
        y = title_height + pad + r * (th + label_height + pad)

        # 缩放并居中贴入格子
        im_resized = im.resize((tw, th))
        canvas.paste(im_resized, (x, y))

        if valid_labels and idx < len(valid_labels):
            ly = y + th + int(line_height * 0.2)
            for line in label_lines_list[idx]:
                draw.text((x + 4, ly), line, fill=(0, 0, 0), font=font)
                ly += line_height

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    canvas.save(output_path, dpi=(300, 300))
    print(f"✅ 图片已整合为一张图：{output_path}")
    return output_path
