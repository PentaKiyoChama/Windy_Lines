#ifndef SDK_PROC_AMP
#define SDK_PROC_AMP

#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#include "PrGPU/KernelSupport/KernelMemory.h"

// Metal specific includes
#if GF_DEVICE_TARGET_METAL
#include <metal_stdlib>
#include <metal_atomic>
		using namespace metal;
#endif

#if GF_DEVICE_TARGET_DEVICE
		uint HashUInt(uint x)
		{
			x ^= x >> 16;
			x *= 0x7feb352d;
			x ^= x >> 15;
			x *= 0x846ca68b;
			x ^= x >> 16;
			return x;
		}

		float Rand01(uint x)
		{
			return (float)(HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
		}

		float EaseInOutSine(float t)
		{
			return 0.5f * (1.0f - cos(3.14159265f * t));
		}

		float DepthScale(float depth, float strength)
		{
			// Shrink lines based on depth: depth=1 (front) keeps scale=1.0, depth=0 (back) shrinks
			float v = 1.0f - (1.0f - depth) * strength;
			return v < 0.05f ? 0.05f : v;
		}

		float ApplyEasing(float t, int easing)
		{
			if (t < 0.0f) t = 0.0f;
			if (t > 1.0f) t = 1.0f;
			if (easing == 0)
			{
				return t;
			}
			else if (easing == 1)
			{
				return 1.0f - cos(3.14159265f * t * 0.5f);
			}
			else if (easing == 2)
			{
				return sin(3.14159265f * t * 0.5f);
			}
			else if (easing == 3)
			{
				return EaseInOutSine(t);
			}
			else if (easing == 4)
			{
				return t * t;
			}
			else if (easing == 5)
			{
				return 1.0f - (1.0f - t) * (1.0f - t);
			}
			else if (easing == 6)
			{
				return t < 0.5f ? 2.0f * t * t : 1.0f - pow(-2.0f * t + 2.0f, 2.0f) * 0.5f;
			}
			else if (easing == 7)
			{
				return t * t * t;
			}
			else if (easing == 8)
			{
				float u = 1.0f - t;
				return 1.0f - u * u * u;
			}
			else if (easing == 9)
			{
				return t < 0.5f ? 4.0f * t * t * t : 1.0f - pow(-2.0f * t + 2.0f, 3.0f) * 0.5f;
			}
			return t;
		}
#	endif  // GF_DEVICE_TARGET_DEVICE

#if GF_DEVICE_TARGET_OPENCL
		inline void AtomicMinInt(volatile __global int* addr, int value)
		{
			atomic_min(addr, value);
		}

		inline void AtomicMaxInt(volatile __global int* addr, int value)
		{
			atomic_max(addr, value);
		}
#endif

#if GF_DEVICE_TARGET_METAL
		// Metal 1.1: Use atomic_int type from metal_atomic
		inline void AtomicMinInt(volatile device atomic_int* addr, int value)
		{
			// Emulate atomic_min using compare-and-swap
			int old_val = atomic_load_explicit(addr, memory_order_relaxed);
			while (old_val > value)
			{
				if (atomic_compare_exchange_weak_explicit(addr, &old_val, value, 
					memory_order_relaxed, memory_order_relaxed))
				{
					break;
				}
			}
		}

		inline void AtomicMaxInt(volatile device atomic_int* addr, int value)
		{
			// Emulate atomic_max using compare-and-swap
			int old_val = atomic_load_explicit(addr, memory_order_relaxed);
			while (old_val < value)
			{
				if (atomic_compare_exchange_weak_explicit(addr, &old_val, value, 
					memory_order_relaxed, memory_order_relaxed))
				{
					break;
				}
			}
		}
#endif

#if GF_DEVICE_TARGET_OPENCL || GF_DEVICE_TARGET_METAL
		GF_KERNEL_FUNCTION(AlphaBoundsKernel,
			((GF_PTR(float4))(ioImage))
			((GF_PTR(int))(outBounds)),
			((int)(inPitch))
			((int)(in16f))
			((unsigned int)(inWidth))
			((unsigned int)(inHeight))
			((int)(inStride))
			((float)(inThreshold)),
			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < inWidth && inXY.y < inHeight)
			{
				if ((int)(inXY.x % (uint)inStride) != 0 || (int)(inXY.y % (uint)inStride) != 0)
				{
					return;
				}
				const float4 pixel = ReadFloat4(ioImage, inXY.y * inPitch + inXY.x, !!in16f);
				if (pixel.w > inThreshold)
				{
#if GF_DEVICE_TARGET_OPENCL
					AtomicMinInt((volatile __global int*)&outBounds[0], (int)inXY.x);
					AtomicMinInt((volatile __global int*)&outBounds[1], (int)inXY.y);
					AtomicMaxInt((volatile __global int*)&outBounds[2], (int)inXY.x);
					AtomicMaxInt((volatile __global int*)&outBounds[3], (int)inXY.y);
#elif GF_DEVICE_TARGET_METAL
					device atomic_int* bounds = (device atomic_int*)outBounds;
					AtomicMinInt(&bounds[0], (int)inXY.x);
					AtomicMinInt(&bounds[1], (int)inXY.y);
					AtomicMaxInt(&bounds[2], (int)inXY.x);
					AtomicMaxInt(&bounds[3], (int)inXY.y);
#endif
				}
			}
		}

		GF_KERNEL_FUNCTION(ProcAmp2Kernel,
			((GF_PTR(float4))(ioImage))
			((GF_PTR(float4))(inLineData))
			((GF_PTR(int))(inTileOffsets))
			((GF_PTR(int))(inTileCounts))
			((GF_PTR(int))(inLineIndices)),
			((int)(inPitch))
			((int)(in16f))
			((unsigned int)(inWidth))
			((unsigned int)(inHeight))
		((float)(inLineCenterX))
		((float)(inLineCenterY))
		((float)(inOriginOffsetX))
		((float)(inOriginOffsetY))
		((float)(inLineCos))
		((float)(inLineSin))
	((float)(inLineLength))
	((float)(inLineThickness))
	((float)(inLineLifetime))
	((float)(inLineTravel))
		((float)(inLineTailFade))
		((float)(inLineDepthStrength))
		((float)(inLineR))
		((float)(inLineG))
		((float)(inLineB))
		((float)(inLineAA))
		((int)(inLineCap))
		((int)(inLineCount))
		((int)(inLineSeed))
		((int)(inLineEasing))
		((int)(inLineInterval))
		((int)(inLineAllowMidPlay))
		((int)(inHideElement))
		((int)(inBlendMode))
		((float)(inFrameIndex))
		((int)(inLineDownsample))
		((int)(inTileCountX))
		((int)(inTileSize))
		((int)(inFocusEnable))
		((float)(inFocusDepth))
		((float)(inFocusRange))
		((float)(inFocusBlurStrength))
		((int)(inShadowEnable))
		((float)(inShadowColorR))
		((float)(inShadowColorG))
		((float)(inShadowColorB))
		((float)(inShadowOffsetX))
		((float)(inShadowOffsetY))
		((float)(inShadowOpacity))
		((float)(inSpawnScaleX))
		((float)(inSpawnScaleY))
		((float)(inSpawnRotationCos))
		((float)(inSpawnRotationSin))
		((int)(inShowSpawnArea))
		((float)(inSpawnAreaColorR))
		((float)(inSpawnAreaColorG))
		((float)(inSpawnAreaColorB))
		((int)(inIsBGRA))
		((float)(inAlphaBoundsMinX))
		((float)(inAlphaBoundsMinY))
		((float)(inAlphaBoundsWidth))
		((float)(inAlphaBoundsHeight))
		((int)(inMotionBlurEnable))
		((int)(inMotionBlurSamples))
		((float)(inMotionBlurStrength))
		((float)(inMotionBlurVelocity)),
			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < inWidth && inXY.y < inHeight)
			{
				float4 pixel = ReadFloat4(ioImage, inXY.y * inPitch + inXY.x, !!in16f);
				
				// Hide original element - start with transparent black
				if (inHideElement != 0)
				{
					pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
				}
				
				const float originalAlpha = pixel.w;
				const float originalR = pixel.x;  // Save original color for Alpha XOR mode
				const float originalG = pixel.y;
				const float originalB = pixel.z;
				const float aa = inLineAA > 0.0f ? inLineAA : 1.0f;
				
				// Front lines accumulator for blend mode 2
				float frontR = 0.0f, frontG = 0.0f, frontB = 0.0f, frontA = 0.0f;
				float frontAppearAlpha = 1.0f;
				
				// Track line-only alpha for Alpha XOR mode (blend mode 3)
				float lineOnlyAlpha = 0.0f;
				
				// Safe tile-based access
				int count = 0;
				int start = 0;
				
				if (inTileSize > 0 && inTileCountX > 0)
				{
					int tileX = (int)(inXY.x / (unsigned int)inTileSize);
					int tileY = (int)(inXY.y / (unsigned int)inTileSize);
					int tileIndex = tileY * inTileCountX + tileX;
					start = inTileOffsets[tileIndex];
					count = inTileCounts[tileIndex];
				}
				
				// Main line pass
				for (int j = 0; j < count; ++j)
				{
					int lineIndex = inLineIndices[start + j];
					int base = lineIndex * 4;  // 4 float4s per line: d0, d1, d2, d3
					float4 d0 = inLineData[base];
					float4 d1 = inLineData[base + 1];
					float4 d2 = inLineData[base + 2];
					float4 d3 = inLineData[base + 3];  // Extra data (appearAlpha)
					
					float centerX = d0.x;
					float centerY = d0.y;
					float lineCos = d0.z;
					float lineSin = d0.w;
					float halfLen = d1.x;
					float halfThick = d1.y;
					float segCenterX = d1.z;
					float lineDepthValue = d1.w;
					float appearAlpha = d3.x;  // Appear/disappear fade alpha
					
					// Depth scaling
					float depthScale = DepthScale(lineDepthValue, inLineDepthStrength);
					float fadeStart = 0.6f;
					float fadeEnd = 0.2f;
					float tDepth = fmin(fmax((depthScale - fadeEnd) / (fadeStart - fadeEnd), 0.0f), 1.0f);
					float depthAlpha = 0.05f + (1.0f - 0.05f) * tDepth;
					
					// Line color and velocity from data
					float lineColorR = d2.x;
					float lineColorG = d2.y;
					float lineColorB = d2.z;
					float lineVelocity = d2.w;  // Instantaneous velocity (0-2 range typically)
					
					// Main line drawing - calculate dx, dy for both line and shadow
					float dx = (float)inXY.x + 0.5f - centerX;
					float dy = (float)inXY.y + 0.5f - centerY;
					
					// Shadow uses offset position
					float sdx = dx - inShadowOffsetX;
					float sdy = dy - inShadowOffsetY;
					
					float coverage = 0.0f;
					float shadowCoverage = 0.0f;
					
					// ========================================
					// NEW Motion Blur Implementation (v2)
					// ========================================
					// Simple approach:
					// - Line moves in +X direction (local coords)
					// - Trail = render line at PAST positions (where it came from)
					// - Past positions have SMALLER X values
					// - To render past: shift LINE position backward = shift PIXEL position forward
					// ========================================
					
					if (inMotionBlurEnable != 0 && inMotionBlurStrength > 0.0f && inMotionBlurSamples > 1)
					{
						float shutterFraction = inMotionBlurStrength / 360.0f;
						float pixelsPerFrame = inLineTravel / inLineLifetime;
						float effectiveVelocity = pixelsPerFrame * lineVelocity;
						float blurLength = effectiveVelocity * shutterFraction;
						
						int samples = inMotionBlurSamples;
						
						if (blurLength > 0.5f)
						{
							float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
							
							// Transform pixel to line's local coordinate system ONCE
							float pxBase = dx * lineCos + dy * lineSin - segCenterX;
							float py = -dx * lineSin + dy * lineCos;
							
							// Shadow base position
							float spxBase = sdx * lineCos + sdy * lineSin - segCenterX;
							float spy = -sdx * lineSin + sdy * lineCos;
							
							float accumA = 0.0f;
							float saccumA = 0.0f;
							
							for (int s = 0; s < samples; ++s)
							{
								// t: -0.5 to +0.5 for centered bidirectional blur
								// t=-0.5: behind (past), t=0: current, t=+0.5: ahead (future)
								float t = ((float)s / fmax((float)(samples - 1), 1.0f)) - 0.5f;
								
								// Bidirectional blur offset: centered on current position
								float blurOffset = blurLength * t;
								
								// Line sample
								float px = pxBase + blurOffset;
								
								// Distance calculation (SDF)
								float dist = 0.0f;
								if (inLineCap == 0) {
									float dxBox = fabs(px) - halfLen;
									float dyBox = fabs(py) - halfThick;
									float ox = dxBox > 0.0f ? dxBox : 0.0f;
									float oy = dyBox > 0.0f ? dyBox : 0.0f;
									dist = sqrt(ox * ox + oy * oy) + fmin(fmax(dxBox, dyBox), 0.0f);
								} else {
									float ax = fabs(px) - halfLen;
									float qx = ax > 0.0f ? ax : 0.0f;
									dist = sqrt(qx * qx + py * py) - halfThick;
								}
								
								float tailT = fmin(fmax((px + halfLen) / denom, 0.0f), 1.0f);
								float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
								
								float sampleCov = 0.0f;
								if (aa > 0.0f) {
									float tt = fmin(fmax((dist - aa) / (0.0f - aa), 0.0f), 1.0f);
									sampleCov = tt * tt * (3.0f - 2.0f * tt) * tailFade * depthAlpha;
								}
								accumA += sampleCov;
								
								// Shadow sample
								if (inShadowEnable != 0) {
									float spx = spxBase + blurOffset;
									
									float sdist = 0.0f;
									if (inLineCap == 0) {
										float dxBox = fabs(spx) - halfLen;
										float dyBox = fabs(spy) - halfThick;
										float ox = dxBox > 0.0f ? dxBox : 0.0f;
										float oy = dyBox > 0.0f ? dyBox : 0.0f;
										sdist = sqrt(ox * ox + oy * oy) + fmin(fmax(dxBox, dyBox), 0.0f);
									} else {
										float ax = fabs(spx) - halfLen;
										float qx = ax > 0.0f ? ax : 0.0f;
										sdist = sqrt(qx * qx + spy * spy) - halfThick;
									}
									
									float stailT = fmin(fmax((spx + halfLen) / denom, 0.0f), 1.0f);
									float stailFade = 1.0f + (stailT - 1.0f) * inLineTailFade;
									
									float ssampleCov = 0.0f;
									if (aa > 0.0f) {
										float tt = fmin(fmax((sdist - aa) / (0.0f - aa), 0.0f), 1.0f);
										ssampleCov = tt * tt * (3.0f - 2.0f * tt) * stailFade * depthAlpha;
									}
									saccumA += ssampleCov;
								}
							}
							
							// Simple average coverage (equal weight for all samples)
							float lineAlpha = accumA / (float)samples;
							float shadowAlphaFinal = saccumA / (float)samples;
							
							// Apply shadow first
							if (inShadowEnable != 0 && shadowAlphaFinal > 0.0f)
							{
								float shadowBlend = shadowAlphaFinal * inShadowOpacity;
								if (inBlendMode == 0) {
									shadowBlend = shadowBlend * (1.0f - originalAlpha);
								} else if (inBlendMode == 2 && lineDepthValue < 0.5f) {
									shadowBlend = shadowBlend * (1.0f - originalAlpha);
								}
								float invShadow = 1.0f - shadowBlend;
								float outAlpha = shadowBlend + pixel.w * invShadow;
								if (outAlpha > 0.0f) {
									pixel.x = (inShadowColorR * shadowBlend + pixel.x * pixel.w * invShadow) / outAlpha;
									pixel.y = (inShadowColorG * shadowBlend + pixel.y * pixel.w * invShadow) / outAlpha;
									pixel.z = (inShadowColorB * shadowBlend + pixel.z * pixel.w * invShadow) / outAlpha;
								}
								float prevAlphaShadow = pixel.w;
								pixel.w = prevAlphaShadow + (outAlpha - prevAlphaShadow) * appearAlpha;
							}
							
							// Apply line color
							if (lineAlpha > 0.0f)
							{
								float prevAlphaLine = pixel.w;
								float srcAlpha = lineAlpha;
								if (inBlendMode == 0) {
									srcAlpha = lineAlpha * (1.0f - originalAlpha);
								} else if (inBlendMode == 2 && lineDepthValue < 0.5f) {
									srcAlpha = lineAlpha * (1.0f - originalAlpha);
								}
								
								if (inBlendMode == 2 && lineDepthValue >= 0.5f) {
									float invFront = 1.0f - srcAlpha;
									float outA = srcAlpha + frontA * invFront;
									if (outA > 0.0f) {
										frontR = (lineColorR * srcAlpha + frontR * frontA * invFront) / outA;
										frontG = (lineColorG * srcAlpha + frontG * frontA * invFront) / outA;
										frontB = (lineColorB * srcAlpha + frontB * frontA * invFront) / outA;
									}
									frontA = outA;
									frontAppearAlpha = fmin(frontAppearAlpha, appearAlpha);
								}
								else if (inBlendMode == 3) {
									// Alpha XOR: line-to-line uses normal blend (XOR applied after loop)
									float invAlpha = 1.0f - srcAlpha;
									float outAlpha = srcAlpha + pixel.w * invAlpha;
									if (outAlpha > 0.0f) {
										pixel.x = (lineColorR * srcAlpha + pixel.x * pixel.w * invAlpha) / outAlpha;
										pixel.y = (lineColorG * srcAlpha + pixel.y * pixel.w * invAlpha) / outAlpha;
										pixel.z = (lineColorB * srcAlpha + pixel.z * pixel.w * invAlpha) / outAlpha;
									}
									pixel.w = prevAlphaLine + (outAlpha - prevAlphaLine) * appearAlpha;
									// Track line-only alpha
									lineOnlyAlpha = fmax(lineOnlyAlpha, srcAlpha * appearAlpha);
								}
								else {
									float invAlpha = 1.0f - srcAlpha;
									float outAlpha = srcAlpha + pixel.w * invAlpha;
									if (outAlpha > 0.0f) {
										pixel.x = (lineColorR * srcAlpha + pixel.x * pixel.w * invAlpha) / outAlpha;
										pixel.y = (lineColorG * srcAlpha + pixel.y * pixel.w * invAlpha) / outAlpha;
										pixel.z = (lineColorB * srcAlpha + pixel.z * pixel.w * invAlpha) / outAlpha;
									}
									pixel.w = prevAlphaLine + (outAlpha - prevAlphaLine) * appearAlpha;
								}
							}
						}
					}
					
					// No motion blur OR blur too small: single-sample calculation
					if (inMotionBlurEnable == 0 || inMotionBlurStrength <= 0.0f || 
					    (inMotionBlurStrength > 0.0f && ((inLineTravel / inLineLifetime) * lineVelocity * (inMotionBlurStrength / 360.0f)) <= 0.5f))
					{
						// No motion blur: single-sample calculation
						float px = dx * lineCos + dy * lineSin;
						float py = -dx * lineSin + dy * lineCos;
						px -= segCenterX;
						
						float dist = 0.0f;
						if (inLineCap == 0) {
							float dxBox = fabs(px) - halfLen;
							float dyBox = fabs(py) - halfThick;
							float ox = dxBox > 0.0f ? dxBox : 0.0f;
							float oy = dyBox > 0.0f ? dyBox : 0.0f;
							float outside = sqrt(ox * ox + oy * oy);
							float inside = fmin(fmax(dxBox, dyBox), 0.0f);
							dist = outside + inside;
						} else {
							float ax = fabs(px) - halfLen;
							float qx = ax > 0.0f ? ax : 0.0f;
							dist = sqrt(qx * qx + py * py) - halfThick;
						}
						
						float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
						float tailT = fmin(fmax((px + halfLen) / denom, 0.0f), 1.0f);
						float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
						if (aa > 0.0f) {
							float tt = fmin(fmax((dist - aa) / (0.0f - aa), 0.0f), 1.0f);
							coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * depthAlpha;
						}
						
						// Shadow: same calculation with offset position
						if (inShadowEnable != 0) {
							float spx = sdx * lineCos + sdy * lineSin;
							float spy = -sdx * lineSin + sdy * lineCos;
							spx -= segCenterX;
							
							float sdist = 0.0f;
							if (inLineCap == 0) {
								float dxBox = fabs(spx) - halfLen;
								float dyBox = fabs(spy) - halfThick;
								float ox = dxBox > 0.0f ? dxBox : 0.0f;
								float oy = dyBox > 0.0f ? dyBox : 0.0f;
								float outside = sqrt(ox * ox + oy * oy);
								float inside = fmin(fmax(dxBox, dyBox), 0.0f);
								sdist = outside + inside;
							} else {
								float ax = fabs(spx) - halfLen;
								float qx = ax > 0.0f ? ax : 0.0f;
								sdist = sqrt(qx * qx + spy * spy) - halfThick;
							}
							
							float stailT = fmin(fmax((spx + halfLen) / denom, 0.0f), 1.0f);
							float stailFade = 1.0f + (stailT - 1.0f) * inLineTailFade;
							if (aa > 0.0f) {
								float tt = fmin(fmax((sdist - aa) / (0.0f - aa), 0.0f), 1.0f);
								shadowCoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * depthAlpha;
							}
						}
					
						// Draw shadow first (before line) using the same coverage calculation
						if (inShadowEnable != 0 && shadowCoverage > 0.0f)
						{
							float scoverage = shadowCoverage * inShadowOpacity;
							float shadowBlend = scoverage;
							if (inBlendMode == 0) {
								shadowBlend = scoverage * (1.0f - originalAlpha);
							} else if (inBlendMode == 2 && lineDepthValue < 0.5f) {
								shadowBlend = scoverage * (1.0f - originalAlpha);
							}
							pixel.x = pixel.x + (inShadowColorR - pixel.x) * shadowBlend;
							pixel.y = pixel.y + (inShadowColorG - pixel.y) * shadowBlend;
							pixel.z = pixel.z + (inShadowColorB - pixel.z) * shadowBlend;
							pixel.w = fmax(pixel.w, shadowBlend);
						}
						
						// Apply blend mode
						if (coverage > 0.0f)
						{
							// Save alpha before this line's contribution
							float prevAlpha = pixel.w;
							
							if (inBlendMode == 0) { // Back
								float backBlend = coverage * (1.0f - originalAlpha);
								pixel.x = pixel.x + (lineColorR - pixel.x) * backBlend;
								pixel.y = pixel.y + (lineColorG - pixel.y) * backBlend;
								pixel.z = pixel.z + (lineColorB - pixel.z) * backBlend;
								float newAlpha = fmax(prevAlpha, backBlend);
								pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
							}
							else if (inBlendMode == 1) { // Front
								pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
								pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
								pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
								float newAlpha = fmax(prevAlpha, coverage);
								pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
							}
							else if (inBlendMode == 2) { // Back and Front
								if (lineDepthValue < 0.5f) {
									float backBlend = coverage * (1.0f - originalAlpha);
									pixel.x = pixel.x + (lineColorR - pixel.x) * backBlend;
									pixel.y = pixel.y + (lineColorG - pixel.y) * backBlend;
									pixel.z = pixel.z + (lineColorB - pixel.z) * backBlend;
									float newAlpha = fmax(prevAlpha, backBlend);
									pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
								} else {
									float aFront = coverage;
									float premR = lineColorR * aFront;
									float premG = lineColorG * aFront;
									float premB = lineColorB * aFront;
									frontR = premR + frontR * (1.0f - aFront);
									frontG = premG + frontG * (1.0f - aFront);
									frontB = premB + frontB * (1.0f - aFront);
									frontA = aFront + frontA * (1.0f - aFront);
									frontAppearAlpha = fmin(frontAppearAlpha, appearAlpha);
								}
							}
							else if (inBlendMode == 3) { // Alpha (XOR with original element only)
								// Line-to-line blending: normal Front mode (additive)
								pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
								pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
								pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
								// Normal alpha blend between lines (like Front mode)
								float newAlpha = fmax(prevAlpha, coverage);
								pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
								// Track line-only alpha
								lineOnlyAlpha = fmax(lineOnlyAlpha, coverage * appearAlpha);
							}
						}
					}
				}  // End line loop
				
				// Apply XOR with original element AFTER all lines are drawn (blend mode 3)
				if (inBlendMode == 3 && originalAlpha > 0.0f)
				{
					// Alpha XOR mode: 
					// - Where lines exist: XOR alpha with original, show original color
					// - Where only original exists (no lines): show original element as-is
					
					if (lineOnlyAlpha > 0.0f)
					{
						// XOR alpha calculation: where overlap exists, alpha is reduced
						float xorAlpha = fmin(fmax(originalAlpha + lineOnlyAlpha - (originalAlpha * lineOnlyAlpha * 2.0f), 0.0f), 1.0f);
						
						// For color: where original element exists, show original color
						pixel.x = pixel.x * (1.0f - originalAlpha) + originalR * originalAlpha;
						pixel.y = pixel.y * (1.0f - originalAlpha) + originalG * originalAlpha;
						pixel.z = pixel.z * (1.0f - originalAlpha) + originalB * originalAlpha;
						pixel.w = xorAlpha;
					}
					else
					{
						// No lines here, but original element exists - show original element
						pixel.x = originalR;
						pixel.y = originalG;
						pixel.z = originalB;
						pixel.w = originalAlpha;
					}
				}
				
				// Apply front lines (blend mode 2)
				if (inBlendMode == 2 && frontA > 0.0f)
				{
					float prevAlpha = pixel.w;
					pixel.x = frontR + pixel.x * (1.0f - frontA);
					pixel.y = frontG + pixel.y * (1.0f - frontA);
					pixel.z = frontB + pixel.z * (1.0f - frontA);
					float newAlpha = frontA + prevAlpha * (1.0f - frontA);
					pixel.w = prevAlpha + (newAlpha - prevAlpha) * frontAppearAlpha;
				}
				
				// Draw spawn area preview
				if (inShowSpawnArea != 0)
				{
					float alphaCenterX = inAlphaBoundsMinX + inAlphaBoundsWidth * 0.5f;
					float alphaCenterY = inAlphaBoundsMinY + inAlphaBoundsHeight * 0.5f;
					float halfW = inAlphaBoundsWidth * inSpawnScaleX * 0.5f;
					float halfH = inAlphaBoundsHeight * inSpawnScaleY * 0.5f;
					
					float relX = (float)inXY.x + 0.5f - alphaCenterX - inOriginOffsetX;
					float relY = (float)inXY.y + 0.5f - alphaCenterY - inOriginOffsetY;
					float localX = relX * inSpawnRotationCos + relY * inSpawnRotationSin;
					float localY = -relX * inSpawnRotationSin + relY * inSpawnRotationCos;
					
					if (fabs(localX) <= halfW && fabs(localY) <= halfH)
					{
						float blendAlpha = 0.5f;
						float baseX = (pixel.w <= 0.0f) ? inSpawnAreaColorR : pixel.x;
						float baseY = (pixel.w <= 0.0f) ? inSpawnAreaColorG : pixel.y;
						float baseZ = (pixel.w <= 0.0f) ? inSpawnAreaColorB : pixel.z;
						pixel.x = baseX + (inSpawnAreaColorR - baseX) * blendAlpha;
						pixel.y = baseY + (inSpawnAreaColorG - baseY) * blendAlpha;
						pixel.z = baseZ + (inSpawnAreaColorB - baseZ) * blendAlpha;
						pixel.w = fmax(pixel.w, blendAlpha);
					}
				}
				
				// DEBUG: Draw ORANGE DIAMOND in top-left corner to indicate OpenCL is being used
				// Shape: Diamond (OpenCL = cross-platform)
				{
					int dx = (int)inXY.x - 20;
					int dy = (int)inXY.y - 20;
					// Diamond shape: |dx| + |dy| <= 15
					int absDx = dx < 0 ? -dx : dx;
					int absDy = dy < 0 ? -dy : dy;
					if (absDx + absDy <= 15)
					{
						pixel.x = 1.0f;   // R = 1 (Orange)
						pixel.y = 0.5f;   // G = 0.5
						pixel.z = 0.0f;   // B = 0
						pixel.w = 1.0f;   // A = 1
			}
		}
		
		// Premultiply alpha for proper Premiere Pro compositing
		pixel.x *= pixel.w;
		pixel.y *= pixel.w;
		pixel.z *= pixel.w;
		
		WriteFloat4(pixel, ioImage, inXY.y * inPitch + inXY.x, !!in16f);
	}
}

#endif  // GF_DEVICE_TARGET_OPENCL || GF_DEVICE_TARGET_METAL
#endif  // SDK_PROC_AMP