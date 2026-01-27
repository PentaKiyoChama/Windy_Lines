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
				// Accumulate front lines separately for blend mode 2
			float frontR = 0.0f;
			float frontG = 0.0f;
			float frontB = 0.0f;
			float frontA = 0.0f;

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
				const int base = lineIndex * 3;  // 3 float4s per line: d0, d1, d2
				const float4 d0 = inLineData[base];
				const float4 d1 = inLineData[base + 1];
				const float4 d2 = inLineData[base + 2];  // Color data

		const float centerX = d0.x;
		const float centerY = d0.y;
				const float lineCos = d0.z;
				const float lineSin = d0.w;
				const float halfLen = d1.x;
				float halfThick = d1.y;
				const float segCenterX = d1.z;
				const float lineDepthValue = d1.w;  // Stored depth value
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
						scoverage = tt * tt * (3.0f - 2.0f * tt) * stailFade * inShadowOpacity * depthAlpha;
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

				const float dx = (float)inXY.x + 0.5f - centerX;
				const float dy = (float)inXY.y + 0.5f - centerY;
				
				// Motion blur: sample multiple positions along the movement direction
				float coverage = 0.0f;
				if (inMotionBlurEnable != 0 && inMotionBlurSamples > 1)
				{
					// Motion blur enabled: sample multiple positions
					const int samples = inMotionBlurSamples;
					const float blurRange = inLineTravel * inMotionBlurStrength / inLineLifetime;
					const float stepSize = blurRange / (float)(samples - 1);
					float totalWeight = 0.0f;
					
					for (int s = 0; s < samples; ++s)
					{
						const float sampleOffset = (s - (samples - 1) * 0.5f) * stepSize;
						
					float pxSample = dx * lineCos + dy * lineSin;
					const float pySample = -dx * lineSin + dy * lineCos;
					pxSample -= (segCenterX + sampleOffset);
					
					float distSample = 0.0f;
					if (inLineCap == 0)  // 0 = Flat (box), 1 = Round (capsule)
					{
						// Box distance (flat caps)
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
						// Capsule distance (rounded caps)
						float ax = fabsf(pxSample) - halfLen;
						float qx = ax > 0.0f ? ax : 0.0f;
						distSample = sqrtf(qx * qx + pySample * pySample) - halfThick;
					}						float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
						float tailT = fminf(fmaxf((pxSample + halfLen) / denom, 0.0f), 1.0f);
						float tailFade = 1.0f + (tailT - 1.0f) * inLineTailFade;
						float sampleCoverage = 0.0f;
						if (aa > 0.0f)
						{
							float tt = fminf(fmaxf((distSample - aa) / (0.0f - aa), 0.0f), 1.0f);
							sampleCoverage = tt * tt * (3.0f - 2.0f * tt) * tailFade;
						}
						
						float normalizedPos = fabsf(s - (samples - 1) * 0.5f) / fmaxf((samples - 1) * 0.5f, 1.0f);
						float weight = 1.0f - normalizedPos * 0.5f;
						
						coverage += sampleCoverage * weight;
						totalWeight += weight;
					}
					
					coverage = (coverage / fmaxf(totalWeight, 1.0f)) * focusAlpha * depthAlpha;
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
						coverage = tt * tt * (3.0f - 2.0f * tt) * tailFade * focusAlpha * depthAlpha;
					}
				}
				if (coverage > 0.0f)
				{
					// Apply blend mode
					if (inBlendMode == 0)  // Back (behind element)
					{
						float backBlend = coverage * (1.0f - originalAlpha);
						pixel.x = pixel.x + (lineColorR - pixel.x) * backBlend;
						pixel.y = pixel.y + (lineColorG - pixel.y) * backBlend;
						pixel.z = pixel.z + (lineColorB - pixel.z) * backBlend;
						pixel.w = fmaxf(pixel.w, backBlend);
					}
					else if (inBlendMode == 1)  // Front (in front of element)
					{
						pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
						pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
						pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
						pixel.w = fmaxf(pixel.w, coverage);
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
							pixel.w = fmaxf(pixel.w, backBlend);
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
						}
					}
				else if (inBlendMode == 3)  // Alpha (XOR transparency)
				{
					// Always draw line color
					pixel.x = pixel.x + (lineColorR - pixel.x) * coverage;
					pixel.y = pixel.y + (lineColorG - pixel.y) * coverage;
					pixel.z = pixel.z + (lineColorB - pixel.z) * coverage;
					// XOR alpha only when overlapping element, otherwise normal blend
					if (originalAlpha > 0.0f)
					{
						pixel.w = fminf(fmaxf(originalAlpha + coverage - (originalAlpha * coverage * 2.0f), 0.0f), 1.0f);
					}
					else
					{
						pixel.w = fmaxf(pixel.w, coverage);
					}
				}
				}
			}

			// Apply front lines after back lines (blend mode 2)
			if (inBlendMode == 2 && frontA > 0.0f)
			{
				pixel.x = frontR + pixel.x * (1.0f - frontA);
				pixel.y = frontG + pixel.y * (1.0f - frontA);
				pixel.z = frontB + pixel.z * (1.0f - frontA);
				pixel.w = frontA + pixel.w * (1.0f - frontA);
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

			WriteFloat4(pixel, ioImage, inXY.y * inPitch + inXY.x, !!in16f);
		}
	}
#endif

#if __NVCC__
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
