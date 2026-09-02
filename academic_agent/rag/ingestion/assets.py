from __future__ import annotations

from pathlib import Path
from typing import Any


def render_pdf_assets(source: Path, document_dir: Path, document_id: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render original/parsed previews and extract embedded PDF images."""
    try:
        import fitz  # type: ignore
    except ImportError:
        return []

    original_dir = document_dir / "pages" / "original"
    parsed_dir = document_dir / "pages" / "parsed"
    images_dir = document_dir / "images"
    for directory in (original_dir, parsed_dir, images_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_blocks: list[dict[str, Any]] = []
    with fitz.open(source) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            matrix = fitz.Matrix(1.35, 1.35)
            original = page.get_pixmap(matrix=matrix, alpha=False)
            (original_dir / f"page-{page_number:04d}.png").write_bytes(original.tobytes("png"))

            overlay_page = pdf.load_page(page_number - 1)
            for block in blocks:
                if block.get("page") != page_number or not block.get("bbox"):
                    continue
                bbox = block["bbox"]
                if len(bbox) != 4:
                    continue
                try:
                    color = (0.1, 0.75, 0.9) if block.get("block_type") == "text" else (0.95, 0.35, 0.2)
                    overlay_page.draw_rect(fitz.Rect(*bbox), color=color, width=1.2, overlay=True)
                except (TypeError, ValueError):
                    continue
            parsed = overlay_page.get_pixmap(matrix=matrix, alpha=False)
            (parsed_dir / f"page-{page_number:04d}.png").write_bytes(parsed.tobytes("png"))

            for image_index, image in enumerate(page.get_images(full=True), start=1):
                try:
                    pixmap = fitz.Pixmap(pdf, image[0])
                    if pixmap.n - pixmap.alpha > 3:
                        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                    image_name = f"page-{page_number:04d}-image-{image_index:03d}.png"
                    image_path = images_dir / image_name
                    pixmap.save(str(image_path))
                    image_blocks.append({
                        "id": f"{document_id}:pdf-image:{page_number}:{image_index}",
                        "block_type": "image",
                        "text": f"Embedded image on page {page_number}",
                        "page": page_number,
                        "bbox": None,
                        "metadata": {"image_path": f"images/{image_name}", "asset_url": f"images/{image_name}"},
                    })
                except (RuntimeError, ValueError):
                    continue
    return image_blocks


def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block.get("block_type", "text")
        text = (block.get("text") or "").strip()
        if not text and kind != "image":
            continue
        if kind == "heading":
            level = int(block.get("metadata", {}).get("level", 2))
            lines.append(f"{'#' * max(1, min(6, level))} {text}")
        elif kind == "image":
            path = block.get("metadata", {}).get("image_path", "")
            lines.append(f"![{text or 'image'}]({path})")
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n" if lines else ""

