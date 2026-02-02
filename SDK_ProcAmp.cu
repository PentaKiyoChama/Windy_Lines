#ifndef SDK_PROCAMP_CU
#define SDK_PROCAMP_CU

#if __CUDACC_VER_MAJOR__ >= 9
	#include <cuda_fp16.h>
#endif

#include "PrGPU/KernelSupport/KernelCore.h"
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
	__device__ __forceinline__ unsigned int HashUInt(unsigned int x)
	{
		x ^= x >> 16;
		x *= 0x7feb352d;
		x ^= x >> 15;
		x *= 0x846ca68b;
		x ^= x >> 16;
		return x;
	}

	__device__ __forceinline__ float Rand01(unsigned int x)
	{
		return (float)(HashUInt(x) & 0x00FFFFFF) / 16777215.0f;
	}

	__device__ __forceinline__ float EaseInOutSine(float t)
	{
		return 0.5f * (1.0f - cosf(3.14159265f * t));
	}

	__device__ __forceinline__ float DepthScale(float depth, float strength)
	{
		// Shrink lines based on depth: depth=1 (front) keeps scale=1.0, depth=0 (back) shrinks
		float v = 1.0f - (1.0f - depth) * strength;
		return v < 0.05f ? 0.05f : v;
	}

	__device__ __forceinline__ float ApplyEasing(float t, int easing)
	{
		if (t < 0.0f) t = 0.0f;
		if (t > 1.0f) t = 1.0f;
		if (easing == 0)
		{
			return t;
		}
		else if (easing == 1)
		{
			return 1.0f - cosf(3.14159265f * t * 0.5f);
		}
		else if (easing == 2)
		{
			return sinf(3.14159265f * t * 0.5f);
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
			return t < 0.5f ? 2.0f * t * t : 1.0f - powf(-2.0f * t + 2.0f, 2.0f) * 0.5f;
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
			return t < 0.5f ? 4.0f * t * t * t : 1.0f - powf(-2.0f * t + 2.0f, 3.0f) * 0.5f;
		}
		return t;
	}

	__device__ __forceinline__ int AtomicMinInt(int* addr, int value)
	{
		return atomicMin(addr, value);
	}

	__device__ __forceinline__ int AtomicMaxInt(int* addr, int value)
	{
		return atomicMax(addr, value);
	}

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
			if ((int)(inXY.x % (unsigned int)inStride) != 0 || (int)(inXY.y % (unsigned int)inStride) != 0)
			{
				return;
			}
			const float4 pixel = ReadFloat4(ioImage, inXY.y * inPitch + inXY.x, !!in16f);
			if (pixel.w > inThreshold)
			{
				AtomicMinInt(&outBounds[0], (int)inXY.x);
				AtomicMinInt(&outBounds[1], (int)inXY.y);
				AtomicMaxInt(&outBounds[2], (int)inXY.x);
				AtomicMaxInt(&outBounds[3], (int)inXY.y);
			}
		}
	}

	GF_KERNEL_FUNCTION(ProcAmp2Kernel,
		((GF_PTR(float4))(ioImage)),
		((int)(inPitch))
		((int)(in16f))
		((int)(inIsBGRA))
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
		((GF_PTR(float4))(inLineData))
		((GF_PTR(int))(inTileOffsets))
		((GF_PTR(int))(inTileCounts))
		((GF_PTR(int))(inLineIndices))
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
		((float)(inAlphaBoundsMinX))
		((float)(inAlphaBoundsMinY))
		((float)(inAlphaBoundsWidth))
		((float)(inAlphaBoundsHeight))
		((int)(inMotionBlurEnable))
		((int)(inMotionBlurSamples))
		((float)(inMotionBlurStrength)),
		((uint2)(inXY)(KERNEL_XY)))
	{
		if (inXY.x < inWidth && inXY.y < inHeight)
		{
			float4 pixel = ReadFloat4(ioImage, inXY.y * inPitch + inXY.x, !!in16f);
			
			// Hide original element - start with transparent black
			if (inHideElement != 0)
			{
				pixel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			}
			
			const float originalAlpha = pixel.w;  // Save original alpha for blend modes
			const float originalR = pixel.x;  // Save original color for Alpha XOR mode
			const float originalG = pixel.y;
			const float originalB = pixel.z;
				// Accumulate front lines separately for blend mode 2
			float frontR = 0.0f;
			float frontG = 0.0f;
			float frontB = 0.0f;
			float frontA = 0.0f;
			float frontAppearAlpha = 1.0f;
			
			// Track line-only alpha for Alpha XOR mode (blend mode 3)
			float lineOnlyAlpha = 0.0f;

			const float aa = inLineAA > 0.0f ? inLineAA : 1.0f;
			const int tileX = inTileSize > 0 ? (int)(inXY.x / (unsigned int)inTileSize) : 0;
			const int tileY = inTileSize > 0 ? (int)(inXY.y / (unsigned int)inTileSize) : 0;
			const int tileIndex = tileY * inTileCountX + tileX;
			const int start = inTileOffsets[tileIndex];
			const int count = inTileCounts[tileIndex];
			
			// Main line pass (with shadow drawn first for each line)
			for (int j = 0; j < count; ++j)
			{
				const int lineIndex = inLineIndices[start + j];
				const int base = lineIndex * 4;  // 4 float4s per line: d0, d1, d2, d3
				const float4 d0 = inLineData[base];
				const float4 d1 = inLineData[base + 1];
				const float4 d2 = inLineData[base + 2];  // Color data
				const float4 d3 = inLineData[base + 3];  // Extra data (appearAlpha)

		const float centerX = d0.x;
		const float centerY = d0.y;
				const float lineCos = d0.z;
				const float lineSin = d0.w;
				const float halfLen = d1.x;
				float halfThick = d1.y;
				const float segCenterX = d1.z;
				const float lineDepthValue = d1.w;  // Stored depth value
				const float appearAlpha = d3.x;  // Appear/disappear fade alpha
				const float depthScale = DepthScale(lineDepthValue, inLineDepthStrength);
				const float fadeStart = 0.6f;
				const float fadeEnd = 0.2f;
				const float t = fminf(fmaxf((depthScale - fadeEnd) / (fadeStart - fadeEnd), 0.0f), 1.0f);
				const float depthAlpha = 0.05f + (1.0f - 0.05f) * t;
				
				// Draw shadow first (before the line)
				if (inShadowEnable != 0)
				{
					const float scenterX = centerX + inShadowOffsetX;
					const float scenterY = centerY + inShadowOffsetY;
					
					const float sdx = (float)inXY.x + 0.5f - scenterX;
					const float sdy = (float)inXY.y + 0.5f - scenterY;
					float spx = sdx * lineCos + sdy * lineSin;
					const float spy = -sdx * lineSin + sdy * lineCos;
				spx -= segCenterX;
				
				float sdist = 0.0f;
				if (inLineCap == 0)  // 0 = Flat (box), 1 = Round (capsule)
				{
					// Box distance (flat caps)
					float dxBox = fabsf(spx) - halfLen;
					float dyBox = fabsf(spy) - halfThick;
					float ox = dxBox > 0.0f ? dxBox : 0.0f;
					float oy = dyBox > 0.0f ? dyBox : 0.0f;
					float outside = sqrtf(ox * ox + oy * oy);
					float inside = fminf(fmaxf(dxBox, dyBox), 0.0f);
					sdist = outside + inside;
				}
				else
				{
					// Capsule distance (rounded caps)
					float ax = fabsf(spx) - halfLen;
					float qx = ax > 0.0f ? ax : 0.0f;
					sdist = sqrtf(qx * qx + spy * spy) - halfThick;
				}					float sdenom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
					float stailT = fminf(fmaxf((spx + halfLen) / sdenom, 0.0f), 1.0f);
					float stailFade = 1.0f + (stailT - 1.0f) * inLineTailFade;
					float scoverage = 0.0f;
					if (aa > 0.0f)
					{
						float tt = fminf(fmaxf((sdist - aa) / (0.0f - aa), 0.0f), 1.0f);
						float smoothCov = tt * tt * (3.0f - 2.0f * tt);
						scoverage = smoothCov * smoothCov * stailFade * inShadowOpacity * depthAlpha;  // Square for sharper edges
					}
					if (scoverage > 0.0f)
					{
						float shadowBlend = scoverage;
						if (inBlendMode == 0)
						{
							// Back: keep shadow behind the original element
							shadowBlend = scoverage * (1.0f - originalAlpha);
						}
						else if (inBlendMode == 2 && lineDepthValue < 0.5f)
						{
							// Back portion of "Back and Front": keep shadow behind the original element
							shadowBlend = scoverage * (1.0f - originalAlpha);
						}

						// Shadow: blend toward pre-converted shadow color
						// inShadowColorR/G/B are already in output format (VUYA or BGRA)
						pixel.x = pixel.x + (inShadowColorR - pixel.x) * shadowBlend;
						pixel.y = pixel.y + (inShadowColorG - pixel.y) * shadowBlend;
						pixel.z = pixel.z + (inShadowColorB - pixel.z) * shadowBlend;
						pixel.w = fmaxf(pixel.w, shadowBlend);
					}
				}
				
				// Focus (Depth of Field) disabled
				float focusAlpha = 1.0f;
				// d2 contains line color (already in output color space)
				const float lineColorR = d2.x;
				const float lineColorG = d2.y;
				const float lineColorB = d2.z;
				const float lineVelocity = d2.w;  // Instantaneous velocity from easing (0-2 range typically)

				const float dx = (float)inXY.x + 0.5f - centerX;
				const float dy = (float)inXY.y + 0.5f - centerY;
				
				// Motion blur: sample multiple positions along the movement direction
				float coverage = 0.0f;
				if (inMotionBlurEnable != 0 && inMotionBlurSamples > 1)
				{
					const int samples = inMotionBlurSamples;
					// Shutter angle based motion blur calculation
					// inMotionBlurStrength is shutter angle (0-360 degrees)
					// Standard film: 180° = half frame exposure
					const float shutterFraction = inMotionBlurStrength / 360.0f;
					const float pixelsPerFrame = inLineTravel / inLineLifetime;
					const float effectiveVelocity = pixelsPerFrame * lineVelocity;
					const float blurRange = effectiveVelocity * shutterFraction;
					const float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
					
					if (blurRange > 0.5f)
					{
						// Standard motion blur: uniform temporal sampling with averaging
						// This matches OpenCL/Metal implementation
						float accumA = 0.0f;
						
						for (int s = 0; s < samples; ++s)
						{
							// Calculate sample offset (evenly distributed)
							// t: 0 -> current position, 1 -> furthest back
							const float t = (float)s / fmaxf((float)(samples - 1), 1.0f);
							
							// Trail mode: blur extends BEHIND the line only
							// t=0 -> current position, t=1 -> furthest back (oldest position)
							// sampleOffset increases as we go further back in time
							const float sampleOffset = blurRange * t;
							
							// Transform pixel to line's local coordinate system
							float pxSample = dx * lineCos + dy * lineSin;
							const float pySample = -dx * lineSin + dy * lineCos;
							// For trail: sample past positions (line was at smaller X values before)
							// segCenterX + sampleOffset shifts the reference point forward
							pxSample -= (segCenterX + sampleOffset);
							
							float distSample = 0.0f;
							if (inLineCap == 0)
							{
								float dxBox = fabsf(pxSample) - halfLen;
								float dyBox = fabsf(pySample) - halfThick;
								float ox = dxBox > 0.0f ? dxBox : 0.0f;
								float oy = dyBox > 0.0f ? dyBox : 0.0f;
								float outside = sqrtf(ox * ox + oy * oy);
								float inside = fminf(fmaxf(dxBox, dyBox), 0.0f);
								distSample = outside + inside;
							}
							else
							{
								float ax = fabsf(pxSample) - halfLen;
								float qx = ax > 0.0f ? ax : 0.0f;
								distSample = sqrtf(qx * qx + pySample * pySample) - halfThick;
							}
							
							float tailT = fminf(fmaxf((pxSample + halfLen) / denom, 0.0f), 1.0f);
							float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
							float sampleCoverage = 0.0f;
							if (aa > 0.0f)
							{
								float tt = fminf(fmaxf((distSample - aa) / (0.0f - aa), 0.0f), 1.0f);
								float smoothCov = tt * tt * (3.0f - 2.0f * tt);
								sampleCoverage = smoothCov * smoothCov * tailFade;  // Square for sharper edges
							}
							
							// Uniform averaging (standard motion blur)
							accumA += sampleCoverage;
						}
						
						// Average the coverage
						coverage = (accumA / (float)samples) * focusAlpha * depthAlpha;
					}
					else
					{
						// Blur too small, use single sample
						float px = dx * lineCos + dy * lineSin;
						const float py = -dx * lineSin + dy * lineCos;
						px -= segCenterX;
						
						float dist = 0.0f;
						if (inLineCap == 0)
						{
							float dxBox = fabsf(px) - halfLen;
							float dyBox = fabsf(py) - halfThick;
							float ox = dxBox > 0.0f ? dxBox : 0.0f;
							float oy = dyBox > 0.0f ? dyBox : 0.0f;
							float outside = sqrtf(ox * ox + oy * oy);
							float inside = fminf(fmaxf(dxBox, dyBox), 0.0f);
							dist = outside + inside;
						}
						else
						{
							float ax = fabsf(px) - halfLen;
							float qx = ax > 0.0f ? ax : 0.0f;
							dist = sqrtf(qx * qx + py * py) - halfThick;
						}
						
						float tailT = fminf(fmaxf((px + halfLen) / denom, 0.0f), 1.0f);
						float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
						if (aa > 0.0f)
						{
							float tt = fminf(fmaxf((dist - aa) / (0.0f - aa), 0.0f), 1.0f);
							float smoothCov = tt * tt * (3.0f - 2.0f * tt);
							coverage = smoothCov * smoothCov * tailFade * focusAlpha * depthAlpha;  // Square for sharper edges
						}
					}
				}
				else
				{
				// No motion blur: original single-sample calculation
				float px = dx * lineCos + dy * lineSin;
				const float py = -dx * lineSin + dy * lineCos;
				px -= segCenterX;

				float dist = 0.0f;
				if (inLineCap == 0)  // 0 = Flat (box), 1 = Round (capsule)
				{
					// Box distance (flat caps)
					float dxBox = fabsf(px) - halfLen;
					float dyBox = fabsf(py) - halfThick;
					float ox = dxBox > 0.0f ? dxBox : 0.0f;
					float oy = dyBox > 0.0f ? dyBox : 0.0f;
					float outside = sqrtf(ox * ox + oy * oy);
					float inside = fminf(fmaxf(dxBox, dyBox), 0.0f);
					dist = outside + inside;
				}
				else
				{
					// Capsule distance (rounded caps)
					float ax = fabsf(px) - halfLen;
					float qx = ax > 0.0f ? ax : 0.0f;
					dist = sqrtf(qx * qx + py * py) - halfThick;
				}					float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
					float tailT = fminf(fmaxf((px + halfLen) / denom, 0.0f), 1.0f);
					float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
					if (aa > 0.0f)
					{
						float tt = fminf(fmaxf((dist - aa) / (0.0f - aa), 0.0f), 1.0f);
						float smoothCov = tt * tt * (3.0f - 2.0f * tt);
						coverage = smoothCov * smoothCov * tailFade * focusAlpha * depthAlpha;  // Square for sharper edges
					}
				}
				if (coverage > 0.0f)
				{
					// Save alpha before this line's contribution
					float prevAlpha = pixel.w;
					
					// Apply blend mode
					if (inBlendMode == 0)  // Back (behind element)
					{
						float backBlend = coverage * (1.0f - originalAlpha);
						pixel.x = pixel.x + (lineColorR - pixel.x) * backBlend;
						pixel.y = pixel.y + (lineColorG - pixel.y) * backBlend;
						pixel.z = pixel.z + (lineColorB - pixel.z) * backBlend;
						float newAlpha = fmaxf(prevAlpha, backBlend);
						pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
					}
					else if (inBlendMode == 1)  // Front (in front of element)
					{
						pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
						pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
						pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
						float newAlpha = fmaxf(prevAlpha, coverage);
						pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
					}
					else if (inBlendMode == 2)  // Back and Front (split by per-line depth)
					{
						// Use stored depth value from line data (consistent across frames)
						if (lineDepthValue < 0.5f)
						{
							// Back mode (full) -> apply to pixel immediately
							float backBlend = coverage * (1.0f - originalAlpha);
							pixel.x = pixel.x + (lineColorR - pixel.x) * backBlend;
							pixel.y = pixel.y + (lineColorG - pixel.y) * backBlend;
							pixel.z = pixel.z + (lineColorB - pixel.z) * backBlend;
							float newAlpha = fmaxf(prevAlpha, backBlend);
							pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
						}
						else
						{
							// Front mode (full) -> accumulate separately, apply after loop
							float aFront = coverage;
							float premR = lineColorR * aFront;
							float premG = lineColorG * aFront;
							float premB = lineColorB * aFront;
							frontR = premR + frontR * (1.0f - aFront);
							frontG = premG + frontG * (1.0f - aFront);
							frontB = premB + frontB * (1.0f - aFront);
							frontA = aFront + frontA * (1.0f - aFront);
							frontAppearAlpha = fminf(frontAppearAlpha, appearAlpha);
						}
					}
				else if (inBlendMode == 3)  // Alpha (XOR with original element only)
				{
					// Line-to-line blending: normal Front mode (additive)
					pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
					pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
					pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
					// Normal alpha blend between lines (like Front mode)
					float newAlpha = fmaxf(prevAlpha, coverage);
					pixel.w = prevAlpha + (newAlpha - prevAlpha) * appearAlpha;
					// Track line-only alpha
					lineOnlyAlpha = fmaxf(lineOnlyAlpha, coverage * appearAlpha);
				}
				}
			}

			// Apply XOR with original element AFTER all lines are drawn (blend mode 3)
			if (inBlendMode == 3 && originalAlpha > 0.0f)
			{
				// Alpha XOR mode: 
				// - Where lines exist: XOR alpha with original, show original color
				// - Where only original exists (no lines): show original element as-is
				
				if (lineOnlyAlpha > 0.0f)
				{
					// XOR alpha calculation: where overlap exists, alpha is reduced
					float xorAlpha = fminf(fmaxf(originalAlpha + lineOnlyAlpha - (originalAlpha * lineOnlyAlpha * 2.0f), 0.0f), 1.0f);
					
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

			// Apply front lines after back lines (blend mode 2)
			if (inBlendMode == 2 && frontA > 0.0f)
			{
				float prevAlpha = pixel.w;
				pixel.x = frontR + pixel.x * (1.0f - frontA);
				pixel.y = frontG + pixel.y * (1.0f - frontA);
				pixel.z = frontB + pixel.z * (1.0f - frontA);
				pixel.z = frontB + pixel.z * (1.0f - frontA);
				float newAlpha = frontA + prevAlpha * (1.0f - frontA);
				pixel.w = prevAlpha + (newAlpha - prevAlpha) * frontAppearAlpha;
			}

			// Draw spawn area preview (filled with inverted colors)
			if (inShowSpawnArea != 0)
			{
				const float alphaCenterX = inAlphaBoundsMinX + inAlphaBoundsWidth * 0.5f;
				const float alphaCenterY = inAlphaBoundsMinY + inAlphaBoundsHeight * 0.5f;
				const float halfW = inAlphaBoundsWidth * inSpawnScaleX * 0.5f;
				const float halfH = inAlphaBoundsHeight * inSpawnScaleY * 0.5f;
				
				// Transform pixel position to rotated spawn space
				const float relX = (float)inXY.x + 0.5f - alphaCenterX - inOriginOffsetX;
				const float relY = (float)inXY.y + 0.5f - alphaCenterY - inOriginOffsetY;
				// Inverse rotate to check bounds
				const float localX = relX * inSpawnRotationCos + relY * inSpawnRotationSin;
				const float localY = -relX * inSpawnRotationSin + relY * inSpawnRotationCos;
				
				// Check if inside the spawn area (filled)
				if (fabsf(localX) <= halfW && fabsf(localY) <= halfH)
				{
					// Blend with spawn area color at 50%, then invert
					const float blendAlpha = 0.5f;
					const float baseX = (pixel.w <= 0.0f) ? inSpawnAreaColorR : pixel.x;
					const float baseY = (pixel.w <= 0.0f) ? inSpawnAreaColorG : pixel.y;
					const float baseZ = (pixel.w <= 0.0f) ? inSpawnAreaColorB : pixel.z;
					float blendedX = baseX + (inSpawnAreaColorR - baseX) * blendAlpha;
					float blendedY = baseY + (inSpawnAreaColorG - baseY) * blendAlpha;
					float blendedZ = baseZ + (inSpawnAreaColorB - baseZ) * blendAlpha;
					pixel.x = blendedX;
					pixel.y = blendedY;
					pixel.z = blendedZ;
					pixel.w = fmaxf(pixel.w, blendAlpha);
				}
			}

			// DEBUG: Draw GREEN SQUARE in top-left corner to indicate CUDA is being used
			// Shape: Square (CUDA = NVIDIA)
			if (inXY.x >= 5 && inXY.x < 35 && inXY.y >= 5 && inXY.y < 35)
			{
				pixel.x = 0.0f;   // R = 0
				pixel.y = 1.0f;   // G = 1 (Green for CUDA)
				pixel.z = 0.0f;   // B = 0
			pixel.w = 1.0f;   // A = 1
		}
		
		// Premultiply alpha for proper Premiere Pro compositing
		// If original background was already opaque, keep it opaque to avoid gray edges
		if (originalAlpha >= 0.99f) {
			pixel.w = 1.0f;
		}
		pixel.x *= pixel.w;
		pixel.y *= pixel.w;
		pixel.z *= pixel.w;

		WriteFloat4(pixel, ioImage, inXY.y * inPitch + inXY.x, !!in16f);
	}
}
#endif#if __NVCC__
	void ProcAmp2_CUDA(
		float* ioBuffer,
		int pitch,
		int is16f,
		int isBGRA,
		int width,
		int height,
		float lineCenterX,
		float lineCenterY,
		float originOffsetX,
		float originOffsetY,
		float lineCos,
		float lineSin,
		float lineLength,
		float lineThickness,
		float lineLifetime,
		float lineTravel,
		float lineTailFade,
		float lineDepthStrength,
		float lineR,
		float lineG,
		float lineB,
		float lineAA,
		int lineCap,
		int lineCount,
		int lineSeed,
		int lineEasing,
		int lineInterval,
		int lineAllowMidPlay,
		int hideElement,
		int blendMode,
		float frameIndex,
		int lineDownsample,
		float4* lineData,
		int* tileOffsets,
		int* tileCounts,
		int* lineIndices,
		int tileCountX,
		int tileSize,
		int focusEnable,
		float focusDepth,
		float focusRange,
		float focusBlurStrength,
		int shadowEnable,
		float shadowColorR,
		float shadowColorG,
		float shadowColorB,
		float shadowOffsetX,
		float shadowOffsetY,
		float shadowOpacity,
		float spawnScaleX,
		float spawnScaleY,
		float spawnRotationCos,
		float spawnRotationSin,
		int showSpawnArea,
		float spawnAreaColorR,
		float spawnAreaColorG,
		float spawnAreaColorB,
		float alphaBoundsMinX,
		float alphaBoundsMinY,
		float alphaBoundsWidth,
		float alphaBoundsHeight,
		int motionBlurEnable,
		int motionBlurSamples,
		float motionBlurStrength)
	{
		dim3 blockDim(16, 16, 1);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

		ProcAmp2Kernel<<<gridDim, blockDim, 0>>>(
			(float4*)ioBuffer,
			pitch,
			is16f,
			isBGRA,
			width,
			height,
			lineCenterX,
			lineCenterY,
			originOffsetX,
			originOffsetY,
			lineCos,
			lineSin,
			lineLength,
			lineThickness,
			lineLifetime,
			lineTravel,
			lineTailFade,
			lineDepthStrength,
			lineR,
			lineG,
			lineB,
			lineAA,
			lineCap,
			lineCount,
			lineSeed,
			lineEasing,
			lineInterval,
			lineAllowMidPlay,
			hideElement,
			blendMode,
			frameIndex,
			lineDownsample,
			lineData,
			tileOffsets,
			tileCounts,
			lineIndices,
			tileCountX,
			tileSize,
			focusEnable,
			focusDepth,
			focusRange,
			focusBlurStrength,
			shadowEnable,
			shadowColorR,
			shadowColorG,
			shadowColorB,
			shadowOffsetX,
			shadowOffsetY,
			shadowOpacity,
			spawnScaleX,
			spawnScaleY,
			spawnRotationCos,
			spawnRotationSin,
			showSpawnArea,
			spawnAreaColorR,
			spawnAreaColorG,
			spawnAreaColorB,
			alphaBoundsMinX,
			alphaBoundsMinY,
			alphaBoundsWidth,
			alphaBoundsHeight,
			motionBlurEnable,
			motionBlurSamples,
			motionBlurStrength);

		cudaDeviceSynchronize();
	}

	void ProcAmp2_CUDA_ComputeAlphaBounds(
		float* ioBuffer,
		int pitch,
		int is16f,
		int width,
		int height,
		int* outBounds,
		int stride,
		float alphaThreshold)
	{
		dim3 blockDim(16, 16, 1);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

		AlphaBoundsKernel<<<gridDim, blockDim, 0>>>(
			(float4*)ioBuffer,
			outBounds,
			pitch,
			is16f,
			(unsigned int)width,
			(unsigned int)height,
			stride,
			alphaThreshold);
		cudaDeviceSynchronize();
	}
#endif

#endif
