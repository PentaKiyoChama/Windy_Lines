#ifndef __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___WATERMARK_MASK_H
#define __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___WATERMARK_MASK_H

//
// Placeholder watermark mask — replace with generate_watermark_mask.py output.
//
// Usage:
//   python3 tools/generate_watermark_mask.py \
//     --text "Edit with" \
//     --text2 "おしゃれテロップ・__TPL_MATCH_NAME__" \
//     --font /path/to/NotoSansJP-Regular.ttf \
//     --header-guard __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___WATERMARK_MASK_H \
//     --out __TPL_MATCH_NAME___WatermarkMask.h
//

#include <cstdint>

namespace FreeModeWatermark
{
    constexpr int kMarginX = 16;
    constexpr int kMarginY = 16;
    constexpr int kMaskWidth = 1;
    constexpr int kMaskHeight = 1;
    constexpr float kFillOpacity = 0.66f;
    constexpr float kOutlineOpacity = 0.56f;

static const uint8_t kFillMask[] = {
    255,
};

static const uint8_t kOutlineMask[] = {
    255,
};

    static inline int TextWidthPx()
    {
        return kMaskWidth;
    }

    static inline int TextHeightPx()
    {
        return kMaskHeight;
    }

    static inline uint8_t FillAlphaAt(int x, int y)
    {
        const int lx = x - kMarginX;
        const int ly = y - kMarginY;
        if (lx < 0 || ly < 0 || lx >= kMaskWidth || ly >= kMaskHeight)
        {
            return 0;
        }
        return kFillMask[ly * kMaskWidth + lx];
    }

    static inline uint8_t OutlineAlphaAt(int x, int y)
    {
        const int lx = x - kMarginX;
        const int ly = y - kMarginY;
        if (lx < 0 || ly < 0 || lx >= kMaskWidth || ly >= kMaskHeight)
        {
            return 0;
        }
        return kOutlineMask[ly * kMaskWidth + lx];
    }
}

#endif // __TPL_UPPER_PREFIX_____TPL_UPPER_PLUGIN___WATERMARK_MASK_H
