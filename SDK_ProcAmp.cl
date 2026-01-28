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
		((int)(inMotionBlurType)),
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
				const float aa = inLineAA > 0.0f ? inLineAA : 1.0f;
				
				// Front lines accumulator for blend mode 2
				float frontR = 0.0f, frontG = 0.0f, frontB = 0.0f, frontA = 0.0f;
				
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
					int base = lineIndex * 3;
					float4 d0 = inLineData[base];
					float4 d1 = inLineData[base + 1];
					float4 d2 = inLineData[base + 2];
					
					float centerX = d0.x;
					float centerY = d0.y;
					float lineCos = d0.z;
					float lineSin = d0.w;
					float halfLen = d1.x;
					float halfThick = d1.y;
					float segCenterX = d1.z;
					float lineDepthValue = d1.w;
					
					// Depth scaling
					float depthScale = DepthScale(lineDepthValue, inLineDepthStrength);
					float fadeStart = 0.6f;
					float fadeEnd = 0.2f;
					float tDepth = fmin(fmax((depthScale - fadeEnd) / (fadeStart - fadeEnd), 0.0f), 1.0f);
					float depthAlpha = 0.05f + (1.0f - 0.05f) * tDepth;
					
					// Draw shadow first (before the line)
					if (inShadowEnable != 0)
					{
						float scenterX = centerX + inShadowOffsetX;
						float scenterY = centerY + inShadowOffsetY;
						float sdx = (float)inXY.x + 0.5f - scenterX;
						float sdy = (float)inXY.y + 0.5f - scenterY;
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
						
						float sdenom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
						float stailT = fmin(fmax((spx + halfLen) / sdenom, 0.0f), 1.0f);
						float stailFade = 1.0f + (stailT - 1.0f) * inLineTailFade;
						float scoverage = 0.0f;
						if (aa > 0.0f) {
							float tt = fmin(fmax((sdist - aa) / (0.0f - aa), 0.0f), 1.0f);
							scoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * inShadowOpacity * depthAlpha;
						}
						if (scoverage > 0.0f) {
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
					}  // End shadow
					
					// Line color from data
					float lineColorR = d2.x;
					float lineColorG = d2.y;
					float lineColorB = d2.z;
					
					// Main line drawing
					float dx = (float)inXY.x + 0.5f - centerX;
					float dy = (float)inXY.y + 0.5f - centerY;
					
					float coverage = 0.0f;
					
					// Motion blur
					if (inMotionBlurEnable != 0 && inMotionBlurSamples > 1)
					{
						int samples = inMotionBlurSamples;
						// Blur range from global settings
						float blurRange = inLineTravel * inMotionBlurStrength / inLineLifetime;
						float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
						
						if (inMotionBlurType == 0)
						{
							// Type 0: Bidirectional (both front and back, physically correct)
							// First, calculate current position (s=0) coverage
							float px0 = dx * lineCos + dy * lineSin;
							float py0 = -dx * lineSin + dy * lineCos;
							px0 -= segCenterX;
							
							float dist0 = 0.0f;
							if (inLineCap == 0) {
								float dxBox = fabs(px0) - halfLen;
								float dyBox = fabs(py0) - halfThick;
								float ox = dxBox > 0.0f ? dxBox : 0.0f;
								float oy = dyBox > 0.0f ? dyBox : 0.0f;
								float outside = sqrt(ox * ox + oy * oy);
								float inside = fmin(fmax(dxBox, dyBox), 0.0f);
								dist0 = outside + inside;
							} else {
								float ax = fabs(px0) - halfLen;
								float qx = ax > 0.0f ? ax : 0.0f;
								dist0 = sqrt(qx * qx + py0 * py0) - halfThick;
							}
							
							float tailT0 = fmin(fmax((px0 + halfLen) / denom, 0.0f), 1.0f);
							float tailFade0 = 1.0f + (tailT0 - 1.0f) * inLineTailFade;
							
							float coverage0 = 0.0f;
							if (aa > 0.0f) {
								float tt = fmin(fmax((dist0 - aa) / (0.0f - aa), 0.0f), 1.0f);
								coverage0 = tt * tt * (3.0f - 2.0f * tt) * tailFade0;
							}
							
							// Trail samples: only add coverage BEHIND current line
							float trailCoverage = 0.0f;
							for (int s = 1; s < samples; ++s)
							{
								float trailDist = (float)s / fmax((float)(samples - 1), 1.0f) * blurRange;
								float dxSample = dx + lineCos * trailDist;
								float dySample = dy + lineSin * trailDist;
								
								float pxSample = dxSample * lineCos + dySample * lineSin;
								float pySample = -dxSample * lineSin + dySample * lineCos;
								pxSample -= segCenterX;
								
								float distSample = 0.0f;
								if (inLineCap == 0) {
									float dxBox = fabs(pxSample) - halfLen;
									float dyBox = fabs(pySample) - halfThick;
									float ox = dxBox > 0.0f ? dxBox : 0.0f;
									float oy = dyBox > 0.0f ? dyBox : 0.0f;
									float outside = sqrt(ox * ox + oy * oy);
									float inside = fmin(fmax(dxBox, dyBox), 0.0f);
									distSample = outside + inside;
								} else {
									float ax = fabs(pxSample) - halfLen;
									float qx = ax > 0.0f ? ax : 0.0f;
									distSample = sqrt(qx * qx + pySample * pySample) - halfThick;
								}
								
								float tailT = fmin(fmax((pxSample + halfLen) / denom, 0.0f), 1.0f);
								float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
								float sampleCoverage = 0.0f;
								if (aa > 0.0f) {
									float tt = fmin(fmax((distSample - aa) / (0.0f - aa), 0.0f), 1.0f);
									sampleCoverage = tt * tt * (3.0f - 2.0f * tt) * tailFade;
								}
								
								float trailOnly = fmax(sampleCoverage - coverage0, 0.0f);
								float fade = 1.0f - (float)s / (float)samples * 0.8f;
								trailCoverage = fmax(trailCoverage, trailOnly * fade);
							}
							
							coverage = fmax(coverage0, coverage0 + trailCoverage) * depthAlpha;
						}
						else
						{
							// Type 1: Trail only (behind the line)
							float totalWeight = 0.0f;
							for (int s = 0; s < samples; ++s)
							{
								// Sample from -blurRange/2 to +blurRange/2 (centered)
								float sampleOffset = ((float)s / fmax((float)(samples - 1), 1.0f) - 0.5f) * blurRange;
								float dxSample = dx + lineCos * sampleOffset;
								float dySample = dy + lineSin * sampleOffset;
								
								float pxSample = dxSample * lineCos + dySample * lineSin;
								float pySample = -dxSample * lineSin + dySample * lineCos;
								pxSample -= segCenterX;
								
								float distSample = 0.0f;
								if (inLineCap == 0) {
									float dxBox = fabs(pxSample) - halfLen;
									float dyBox = fabs(pySample) - halfThick;
									float ox = dxBox > 0.0f ? dxBox : 0.0f;
									float oy = dyBox > 0.0f ? dyBox : 0.0f;
									float outside = sqrt(ox * ox + oy * oy);
									float inside = fmin(fmax(dxBox, dyBox), 0.0f);
									distSample = outside + inside;
								} else {
									float ax = fabs(pxSample) - halfLen;
									float qx = ax > 0.0f ? ax : 0.0f;
									distSample = sqrt(qx * qx + pySample * pySample) - halfThick;
								}
								
								float tailT = fmin(fmax((pxSample + halfLen) / denom, 0.0f), 1.0f);
								float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
								float sampleCoverage = 0.0f;
								if (aa > 0.0f) {
									float tt = fmin(fmax((distSample - aa) / (0.0f - aa), 0.0f), 1.0f);
									sampleCoverage = tt * tt * (3.0f - 2.0f * tt) * tailFade;
								}
								
								// Weight: center is strongest, edges fade
								float normalizedPos = fabs((float)s / fmax((float)(samples - 1), 1.0f) - 0.5f) * 2.0f;
								float weight = 1.0f - normalizedPos * 0.5f;
								coverage += sampleCoverage * weight;
								totalWeight += weight;
							}
							coverage = (coverage / fmax(totalWeight, 1.0f)) * depthAlpha;
						}
					}
					else
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
					}  // End motion blur else
					
					// Apply blend mode
					if (coverage > 0.0f)
					{
						if (inBlendMode == 0) { // Back
							float backBlend = coverage * (1.0f - originalAlpha);
							pixel.x = pixel.x + (lineColorR - pixel.x) * backBlend;
							pixel.y = pixel.y + (lineColorG - pixel.y) * backBlend;
							pixel.z = pixel.z + (lineColorB - pixel.z) * backBlend;
							pixel.w = fmax(pixel.w, backBlend);
						}
						else if (inBlendMode == 1) { // Front
							pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
							pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
							pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
							pixel.w = fmax(pixel.w, coverage);
						}
						else if (inBlendMode == 2) { // Back and Front
							if (lineDepthValue < 0.5f) {
								float backBlend = coverage * (1.0f - originalAlpha);
								pixel.x = pixel.x + (lineColorR - pixel.x) * backBlend;
								pixel.y = pixel.y + (lineColorG - pixel.y) * backBlend;
								pixel.z = pixel.z + (lineColorB - pixel.z) * backBlend;
								pixel.w = fmax(pixel.w, backBlend);
							} else {
								float aFront = coverage;
								float premR = lineColorR * aFront;
								float premG = lineColorG * aFront;
								float premB = lineColorB * aFront;
								frontR = premR + frontR * (1.0f - aFront);
								frontG = premG + frontG * (1.0f - aFront);
								frontB = premB + frontB * (1.0f - aFront);
								frontA = aFront + frontA * (1.0f - aFront);
							}
						}
						else if (inBlendMode == 3) { // Alpha (XOR)
							pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
							pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
							pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
							if (originalAlpha > 0.0f) {
								pixel.w = fmin(fmax(originalAlpha + coverage - (originalAlpha * coverage * 2.0f), 0.0f), 1.0f);
							} else {
								pixel.w = fmax(pixel.w, coverage);
							}
						}
					}
				}  // End line loop
				
				// Apply front lines (blend mode 2)
				if (inBlendMode == 2 && frontA > 0.0f)
				{
					pixel.x = frontR + pixel.x * (1.0f - frontA);
					pixel.y = frontG + pixel.y * (1.0f - frontA);
					pixel.z = frontB + pixel.z * (1.0f - frontA);
					pixel.w = frontA + pixel.w * (1.0f - frontA);
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
				
				WriteFloat4(pixel, ioImage, inXY.y * inPitch + inXY.x, !!in16f);
			}
		}

#endif  // GF_DEVICE_TARGET_OPENCL || GF_DEVICE_TARGET_METAL
#endif  // SDK_PROC_AMP
