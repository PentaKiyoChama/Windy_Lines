#ifndef OST_WINDYLINES_COMMON_H
#define OST_WINDYLINES_COMMON_H

#include "PrSDKTypes.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline csSDK_uint32 HashUInt(csSDK_uint32 x)
{
	x ^= x >> 16;
	x *= 0x7feb352d;
	x ^= x >> 15;
	x *= 0x846ca68b;
	x ^= x >> 16;
	return x;
}

static inline float Rand01(csSDK_uint32 x)
{
	return (HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
}

static inline float EaseInOutSine(float t)
{
	return 0.5f * (1.0f - cosf((float)M_PI * t));
}

static inline float DepthScale(float depth, float strength)
{
	const float v = 1.0f - (1.0f - depth) * strength;
	return v < 0.05f ? 0.05f : v;
}

static inline float ApplyEasing(float t, int easingType)
{
	if (t < 0.0f) t = 0.0f;
	if (t > 1.0f) t = 1.0f;
	switch (easingType)
	{
		case 0: return t;
		case 1: return t * t * (3.0f - 2.0f * t);
		case 2: return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
		case 3: return 1.0f - cosf((float)M_PI * t * 0.5f);
		case 4: return sinf((float)M_PI * t * 0.5f);
		case 5: return EaseInOutSine(t);
		case 6: // OutInSine: fast→slow→fast with smooth midpoint
		{
			float outIn;
			if (t < 0.5f) {
				outIn = 0.5f * ApplyEasing(t * 2.0f, 4);
			} else {
				outIn = 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 3);
			}
			const float k = 0.25f;
			return outIn * (1.0f - k) + t * k;
		}
		case 7: return t * t;
		case 8: return 1.0f - (1.0f - t) * (1.0f - t);
		case 9:
		{
			const float u = t * 2.0f;
			if (u < 1.0f) { return 0.5f * u * u; }
			const float v = u - 1.0f;
			return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v));
		}
		case 10: // OutInQuad: fast→slow→fast with smooth midpoint
		{
			float outIn;
			if (t < 0.5f) {
				outIn = 0.5f * ApplyEasing(t * 2.0f, 8);
			} else {
				outIn = 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 7);
			}
			const float k = 0.25f;
			return outIn * (1.0f - k) + t * k;
		}
		case 11: return t * t * t;
		case 12:
		{
			const float u = 1.0f - t;
			return 1.0f - u * u * u;
		}
		case 13:
		{
			const float u = t * 2.0f;
			if (u < 1.0f) { return 0.5f * u * u * u; }
			const float v = u - 1.0f;
			return 0.5f + 0.5f * (1.0f - (1.0f - v) * (1.0f - v) * (1.0f - v));
		}
		case 14: // OutInCubic: fast→slow→fast with smooth midpoint
		{
			float outIn;
			if (t < 0.5f) {
				outIn = 0.5f * ApplyEasing(t * 2.0f, 12);
			} else {
				outIn = 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 11);
			}
			const float k = 0.25f;
			return outIn * (1.0f - k) + t * k;
		}
		case 15: return 1.0f - sqrtf(1.0f - t * t);
		case 16:
		{
			const float u = t - 1.0f;
			return sqrtf(1.0f - u * u);
		}
		case 17:
		{
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return 0.5f * (1.0f - sqrtf(1.0f - u * u));
			}
			const float v = u - 2.0f;
			return 0.5f * (sqrtf(1.0f - v * v) + 1.0f);
		}
		case 18: // OutInCirc: fast→slow→fast with smooth midpoint
		{
			float outIn;
			if (t < 0.5f) {
				outIn = 0.5f * ApplyEasing(t * 2.0f, 16);
			} else {
				outIn = 0.5f + 0.5f * ApplyEasing((t - 0.5f) * 2.0f, 15);
			}
			const float k = 0.25f;
			return outIn * (1.0f - k) + t * k;
		}
		case 19:
		{
			const float s = 1.70158f;
			return t * t * ((s + 1.0f) * t - s);
		}
		case 20:
		{
			const float s = 1.70158f;
			const float u = t - 1.0f;
			return u * u * ((s + 1.0f) * u + s) + 1.0f;
		}
		case 21:
		{
			const float s = 1.70158f * 1.525f;
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return 0.5f * u * u * ((s + 1.0f) * u - s);
			}
			const float v = u - 2.0f;
			return 0.5f * (v * v * ((s + 1.0f) * v + s) + 2.0f);
		}
		case 22:
		{
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.3f;
			return -powf(2.0f, 10.0f * (t - 1.0f)) * sinf((t - 1.0f - p / 4.0f) * (2.0f * (float)M_PI) / p);
		}
		case 23:
		{
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.3f;
			return powf(2.0f, -10.0f * t) * sinf((t - p / 4.0f) * (2.0f * (float)M_PI) / p) + 1.0f;
		}
		case 24:
		{
			if (t == 0.0f) return 0.0f;
			if (t == 1.0f) return 1.0f;
			const float p = 0.45f;
			const float s = p / 4.0f;
			const float u = t * 2.0f;
			if (u < 1.0f) {
				return -0.5f * powf(2.0f, 10.0f * (u - 1.0f)) * sinf((u - 1.0f - s) * (2.0f * (float)M_PI) / p);
			}
			return powf(2.0f, -10.0f * (u - 1.0f)) * sinf((u - 1.0f - s) * (2.0f * (float)M_PI) / p) * 0.5f + 1.0f;
		}
		case 25:
		{
			const float u = 1.0f - t;
			float b;
			if (u < 1.0f / 2.75f) {
				b = 7.5625f * u * u;
			} else if (u < 2.0f / 2.75f) {
				const float v = u - 1.5f / 2.75f;
				b = 7.5625f * v * v + 0.75f;
			} else if (u < 2.5f / 2.75f) {
				const float v = u - 2.25f / 2.75f;
				b = 7.5625f * v * v + 0.9375f;
			} else {
				const float v = u - 2.625f / 2.75f;
				b = 7.5625f * v * v + 0.984375f;
			}
			return 1.0f - b;
		}
		case 26:
		{
			if (t < 1.0f / 2.75f) {
				return 7.5625f * t * t;
			} else if (t < 2.0f / 2.75f) {
				const float u = t - 1.5f / 2.75f;
				return 7.5625f * u * u + 0.75f;
			} else if (t < 2.5f / 2.75f) {
				const float u = t - 2.25f / 2.75f;
				return 7.5625f * u * u + 0.9375f;
			} else {
				const float u = t - 2.625f / 2.75f;
				return 7.5625f * u * u + 0.984375f;
			}
		}
		case 27:
		{
			if (t < 0.5f) {
				const float u = 1.0f - t * 2.0f;
				float b;
				if (u < 1.0f / 2.75f) {
					b = 7.5625f * u * u;
				} else if (u < 2.0f / 2.75f) {
					const float v = u - 1.5f / 2.75f;
					b = 7.5625f * v * v + 0.75f;
				} else if (u < 2.5f / 2.75f) {
					const float v = u - 2.25f / 2.75f;
					b = 7.5625f * v * v + 0.9375f;
				} else {
					const float v = u - 2.625f / 2.75f;
					b = 7.5625f * v * v + 0.984375f;
				}
				return (1.0f - b) * 0.5f;
			} else {
				const float u = t * 2.0f - 1.0f;
				float b;
				if (u < 1.0f / 2.75f) {
					b = 7.5625f * u * u;
				} else if (u < 2.0f / 2.75f) {
					const float v = u - 1.5f / 2.75f;
					b = 7.5625f * v * v + 0.75f;
				} else if (u < 2.5f / 2.75f) {
					const float v = u - 2.25f / 2.75f;
					b = 7.5625f * v * v + 0.9375f;
				} else {
					const float v = u - 2.625f / 2.75f;
					b = 7.5625f * v * v + 0.984375f;
				}
				return b * 0.5f + 0.5f;
			}
		}
		default:
			return t;
	}
}

static inline float ApplyEasingDerivative(float t, int easingType)
{
	const float epsilon = 0.001f;
	switch (easingType)
	{
		case 0: return 1.0f;
		default:
		{
			const float t1 = t > epsilon ? t - epsilon : 0.0f;
			const float t2 = t < 1.0f - epsilon ? t + epsilon : 1.0f;
			const float dt = t2 - t1;
			if (dt > 0.0f) {
				return (ApplyEasing(t2, easingType) - ApplyEasing(t1, easingType)) / dt;
			}
			return 1.0f;
		}
	}
}

static inline float SDFBox(float px, float py, float halfLen, float halfThick)
{
	const float dxBox = fabsf(px) - halfLen;
	const float dyBox = fabsf(py) - halfThick;
	const float ox = fmaxf(dxBox, 0.0f);
	const float oy = fmaxf(dyBox, 0.0f);
	const float outside = sqrtf(ox * ox + oy * oy);
	const float inside = fminf(fmaxf(dxBox, dyBox), 0.0f);
	return outside + inside;
}

static inline float SDFCapsule(float px, float py, float halfLen, float halfThick)
{
	const float ax = fabsf(px) - halfLen;
	const float qx = fmaxf(ax, 0.0f);
	return sqrtf(qx * qx + py * py) - halfThick;
}

static inline void BlendPremultiplied(
	float srcR, float srcG, float srcB, float srcA,
	float& dstR, float& dstG, float& dstB, float& dstA)
{
	const float invSrcA = 1.0f - srcA;
	const float outA = srcA + dstA * invSrcA;
	const float invOutA = 1.0f / fmaxf(outA, 1e-6f);
	dstR = (srcR * srcA + dstR * dstA * invSrcA) * invOutA;
	dstG = (srcG * srcA + dstG * dstA * invSrcA) * invOutA;
	dstB = (srcB * srcA + dstB * dstA * invSrcA) * invOutA;
	dstA = outA;
}

static inline void BlendUnpremultiplied(
	float srcR, float srcG, float srcB, float srcA,
	float& dstR, float& dstG, float& dstB, float& dstA)
{
	const float invSrcA = 1.0f - srcA;
	const float outA = srcA + dstA * invSrcA;
	const float invOutA = 1.0f / fmaxf(outA, 1e-6f);
	dstR = (srcR * srcA + dstR * dstA * invSrcA) * invOutA;
	dstG = (srcG * srcA + dstG * dstA * invSrcA) * invOutA;
	dstB = (srcB * srcA + dstB * dstA * invSrcA) * invOutA;
	dstA = outA;
}

static inline float ApplyLinkageValue(
	float userValue,
	int linkageMode,
	float linkageRate,
	float boundsWidth,
	float boundsHeight,
	float dsScale)
{
	float finalValue = userValue;
	if (linkageMode == LINKAGE_MODE_WIDTH) {
		finalValue = boundsWidth * linkageRate;
	} else if (linkageMode == LINKAGE_MODE_HEIGHT) {
		finalValue = boundsHeight * linkageRate;
	}
	return (linkageMode == LINKAGE_MODE_OFF) ? (finalValue * dsScale) : finalValue;
}

static inline void BuildPresetPalette(float paletteOut[8][3], int presetIndex)
{
	const PresetColor* preset = GetPresetPalette(presetIndex + 1);
	if (!preset)
	{
		return;
	}
	for (int i = 0; i < 8; ++i)
	{
		paletteOut[i][0] = preset[i].r / 255.0f;
		paletteOut[i][1] = preset[i].g / 255.0f;
		paletteOut[i][2] = preset[i].b / 255.0f;
	}
}

#endif