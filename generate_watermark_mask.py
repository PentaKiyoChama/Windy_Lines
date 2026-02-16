#!/usr/bin/env python3
"""
Generate watermark mask header for OST_WindyLines.

Default text: おしゃれテロップ・風を感じるライン
Recommended font: Noto Sans JP (OFL-1.1)

Usage:
  python3 generate_watermark_mask.py \
    --font /path/to/NotoSansJP-Regular.ttf \
    --out /path/to/OST_WindyLines_WatermarkMask.h
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def _measure_segment(
    text: str,
    font: "ImageFont.FreeTypeFont",
    stroke_width: int,
) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) bounding box for a text segment."""
    from PIL import Image, ImageDraw
    dummy = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(dummy)
    return draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)


def _render_segment_pair(
    text1: str, font1: "ImageFont.FreeTypeFont",
    text2: str, font2: "ImageFont.FreeTypeFont",
    stroke_width: int, padding: int, gap: int,
) -> Tuple["Image.Image", "Image.Image", int, int]:
    """
    Render two text segments at different sizes, baseline-aligned,
    and return (fill_img, stroke_img, width, height).
    """
    from PIL import Image, ImageDraw

    l1, t1, r1, b1 = _measure_segment(text1, font1, stroke_width)
    l2, t2, r2, b2 = _measure_segment(text2, font2, stroke_width)

    # Use font metrics for proper baseline alignment
    asc1, desc1 = font1.getmetrics()
    asc2, desc2 = font2.getmetrics()

    max_asc = max(asc1, asc2)
    max_desc = max(desc1, desc2)
    total_height = max_asc + max_desc + padding * 2

    w1 = r1 - l1
    w2 = r2 - l2
    total_width = w1 + gap + w2 + padding * 2

    # Baseline y offset for each segment
    baseline_y1 = padding + (max_asc - asc1)
    baseline_y2 = padding + (max_asc - asc2)

    fill_img = Image.new("L", (total_width, total_height), 0)
    draw_fill = ImageDraw.Draw(fill_img)
    draw_fill.text((padding - l1, baseline_y1 - t1), text1, font=font1, fill=255, stroke_width=0)
    draw_fill.text((padding + w1 + gap - l2, baseline_y2 - t2), text2, font=font2, fill=255, stroke_width=0)

    stroke_img = Image.new("L", (total_width, total_height), 0)
    draw_stroke = ImageDraw.Draw(stroke_img)
    draw_stroke.text((padding - l1, baseline_y1 - t1), text1, font=font1, fill=255, stroke_width=stroke_width)
    draw_stroke.text((padding + w1 + gap - l2, baseline_y2 - t2), text2, font=font2, fill=255, stroke_width=stroke_width)

    return fill_img, stroke_img, total_width, total_height


def render_with_pillow(
    text: str,
    font_path: Path,
    font_size: int,
    stroke_width: int,
    padding: int,
    outline_blur: float,
    binary_threshold: int,
    scale: float,
    text2: str = "",
    font_size2: int = 0,
    gap: int = 6,
) -> Tuple[List[int], List[int], int, int]:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
    except Exception as exc:
        raise RuntimeError("Pillow が必要です。`pip install pillow` を実行してください。") from exc

    if not font_path.exists():
        raise FileNotFoundError(f"フォントが見つかりません: {font_path}")

    font = ImageFont.truetype(str(font_path), font_size)

    if text2:
        # Two-segment mode: render with different sizes, baseline-aligned
        font2 = ImageFont.truetype(str(font_path), font_size2 if font_size2 > 0 else font_size)
        fill_img, stroke_img, width, height = _render_segment_pair(
            text, font, text2, font2, stroke_width, padding, gap)
    else:
        # Single text mode (original behavior)
        dummy = Image.new("L", (8, 8), 0)
        draw = ImageDraw.Draw(dummy)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)

        width = (right - left) + padding * 2
        height = (bottom - top) + padding * 2

        fill_img = Image.new("L", (width, height), 0)
        draw_fill = ImageDraw.Draw(fill_img)
        draw_fill.text(
            (padding - left, padding - top),
            text,
            font=font,
            fill=255,
            stroke_width=0,
        )

        stroke_img = Image.new("L", (width, height), 0)
        draw_stroke = ImageDraw.Draw(stroke_img)
        draw_stroke.text(
            (padding - left, padding - top),
            text,
            font=font,
            fill=255,
            stroke_width=stroke_width,
        )

    # Outline = stroke - fill
    outline_img = Image.new("L", (width, height), 0)
    p_fill = fill_img.load()
    p_stroke = stroke_img.load()
    p_outline = outline_img.load()
    for y in range(height):
        for x in range(width):
            p_outline[x, y] = max(0, p_stroke[x, y] - p_fill[x, y])

    # Soften slightly for nicer edge on YUV blending
    if outline_blur > 0.0:
        outline_img = outline_img.filter(ImageFilter.GaussianBlur(radius=outline_blur))

    if 0 <= binary_threshold <= 255:
        fill_img = fill_img.point(lambda px: 255 if px >= binary_threshold else 0)
        outline_img = outline_img.point(lambda px: 255 if px >= binary_threshold else 0)

    if scale <= 0.0:
        raise ValueError("--scale は 0 より大きい値を指定してください。")

    if abs(scale - 1.0) > 1e-6:
        try:
            from PIL import Image
            resample_nn = Image.Resampling.NEAREST
        except AttributeError:
            resample_nn = Image.NEAREST

        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))
        fill_img = fill_img.resize((scaled_width, scaled_height), resample=resample_nn)
        outline_img = outline_img.resize((scaled_width, scaled_height), resample=resample_nn)
        width = scaled_width
        height = scaled_height

    fill = list(fill_img.getdata())
    outline = list(outline_img.getdata())
    return fill, outline, width, height


def to_cpp_array(name: str, data: List[int], row_width: int) -> str:
    lines = [f"static const uint8_t {name}[] = {{"]
    for i in range(0, len(data), row_width):
        row = data[i : i + row_width]
        body = ", ".join(f"{v:3d}" for v in row)
        lines.append(f"    {body},")
    lines.append("};")
    return "\n".join(lines)


def build_header(fill: List[int], outline: List[int], width: int, height: int, fill_opacity: float, outline_opacity: float) -> str:
    fill_arr = to_cpp_array("kFillMask", fill, width)
    outline_arr = to_cpp_array("kOutlineMask", outline, width)

    return f'''#ifndef OST_WINDYLINES_WATERMARK_MASK_H
#define OST_WINDYLINES_WATERMARK_MASK_H

#include <cstdint>

namespace FreeModeWatermark
{{
    constexpr int kMarginX = 16;
    constexpr int kMarginY = 16;
    constexpr int kMaskWidth = {width};
    constexpr int kMaskHeight = {height};
    constexpr float kFillOpacity = {fill_opacity:.2f}f;
    constexpr float kOutlineOpacity = {outline_opacity:.2f}f;

{fill_arr}

{outline_arr}

    static inline int TextWidthPx()
    {{
        return kMaskWidth;
    }}

    static inline int TextHeightPx()
    {{
        return kMaskHeight;
    }}

    static inline uint8_t FillAlphaAt(int x, int y)
    {{
        const int lx = x - kMarginX;
        const int ly = y - kMarginY;
        if (lx < 0 || ly < 0 || lx >= kMaskWidth || ly >= kMaskHeight)
        {{
            return 0;
        }}
        return kFillMask[ly * kMaskWidth + lx];
    }}

    static inline uint8_t OutlineAlphaAt(int x, int y)
    {{
        const int lx = x - kMarginX;
        const int ly = y - kMarginY;
        if (lx < 0 || ly < 0 || lx >= kMaskWidth || ly >= kMaskHeight)
        {{
            return 0;
        }}
        return kOutlineMask[ly * kMaskWidth + lx];
    }}
}}

#endif // OST_WINDYLINES_WATERMARK_MASK_H
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate watermark mask header from font")
    parser.add_argument("--text", default="Edit with", help="1つ目のテキスト（--text2 指定時は前半部分）")
    parser.add_argument("--text2", default="", help="2つ目のテキスト（別サイズ）。空なら単一テキストモード")
    parser.add_argument("--font", required=True, type=Path, help="Path to TTF/OTF font")
    parser.add_argument("--font-size", type=int, default=28, help="1つ目のテキストのフォントサイズ")
    parser.add_argument("--font-size2", type=int, default=0, help="2つ目のテキストのフォントサイズ（0=--font-sizeと同じ）")
    parser.add_argument("--gap", type=int, default=6, help="text と text2 の間のピクセル間隔")
    parser.add_argument("--stroke-width", type=int, default=2)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--outline-blur", type=float, default=0.4)
    parser.add_argument("--binary-threshold", type=int, default=-1, help="0-255で2値化。-1で無効")
    parser.add_argument("--hard-edge", action="store_true", help="レトロPC風の荒いエッジ（2値化+ぼかし無し）")
    parser.add_argument("--scale", type=float, default=1.0, help="フォントサイズとは独立した最終マスク拡大率")
    parser.add_argument("--opacity", type=float, default=0.66, help="ウォーターマーク不透明度 (0.0-1.0, デフォルト 0.66)")
    parser.add_argument("--out", type=Path, default=Path("OST_WindyLines_WatermarkMask.h"))
    args = parser.parse_args()

    outline_blur = args.outline_blur
    binary_threshold = args.binary_threshold
    if args.hard_edge:
        outline_blur = 0.0
        if binary_threshold < 0:
            binary_threshold = 128

    fill, outline, width, height = render_with_pillow(
        text=args.text,
        font_path=args.font,
        font_size=args.font_size,
        stroke_width=args.stroke_width,
        padding=args.padding,
        outline_blur=outline_blur,
        binary_threshold=binary_threshold,
        scale=args.scale,
        text2=args.text2,
        font_size2=args.font_size2,
        gap=args.gap,
    )

    fill_opacity = max(0.0, min(1.0, args.opacity))
    outline_opacity = round(fill_opacity * 0.85, 2)

    header = build_header(fill, outline, width, height, fill_opacity, outline_opacity)
    args.out.write_text(header, encoding="utf-8")
    print(f"Generated: {args.out}")
    print(f"Size: {width}x{height}")
    print(f"Opacity: fill={fill_opacity:.0%}, outline={outline_opacity:.0%}")


if __name__ == "__main__":
    main()
