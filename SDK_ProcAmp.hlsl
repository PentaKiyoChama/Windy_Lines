/*************************************************************************
 * ADOBE CONFIDENTIAL
 * ___________________
 *
 * Copyright 2023 Adobe
 * All Rights Reserved.
 *
 * NOTICE: All information contained herein is, and remains
 * the property of Adobe and its suppliers, if any. The intellectual
 * and technical concepts contained herein are proprietary to Adobe
 * and its suppliers and are protected by all applicable intellectual
 * property laws, including trade secret and copyright laws.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from Adobe.
 **************************************************************************/
 /*
  * Buffers associated with the shader
  * Read-Write/Write buffers are registered as UAVs while
  * Read-only buffers are registered as SRVs
  * See u0 is bound as UAV in RootSignature below
  */
RWByteAddressBuffer mIOImage: register(u0);
ByteAddressBuffer mLineData: register(t0);
ByteAddressBuffer mTileOffsets: register(t1);
ByteAddressBuffer mTileCounts: register(t2);
ByteAddressBuffer mLineIndices: register(t3);

/*
* Parameters which will be used by the shader
* The structure should exactly match the parameter structure
* in the host code
*/
cbuffer cb : register(b0)
{
	int     mPitch;
    int     mIs16f;
	int     mIsBGRA;
    int     mWidth;
    int     mHeight;
	float   mLineCenterX;
	float   mLineCenterY;
	float   mLineCos;
	float   mLineSin;
	float   mLineLength;
	float   mLineThickness;
	float   mLineLifetime;
	float   mLineTravel;
	float   mLineTailFade;
	float   mLineDepthStrength;
	float   mLineR;
	float   mLineG;
	float   mLineB;
	float   mLineAA;
	int     mLineCap;
	int     mLineCount;
	int     mLineSeed;
	int     mLineEasing;
	int     mLineInterval;
	int     mLineAllowMidPlay;
	int     mHideElement;      // Hide original element (lines only)
	int     mBlendMode;        // Blend mode with element
	int     mShadowEnable;     // Shadow on/off
	float   mShadowColorR;     // Shadow color R (0-1)
	float   mShadowColorG;     // Shadow color G (0-1)
	float   mShadowColorB;     // Shadow color B (0-1)
	float   mShadowOffsetX;    // Shadow offset X (px)
	float   mShadowOffsetY;    // Shadow offset Y (px)
	float   mShadowOpacity;    // Shadow opacity (0-1)
	float   mLineSpawnScaleX;  // Spawn area scale X (0-2, where 1.0 = 100%)
	float   mLineSpawnScaleY;  // Spawn area scale Y (0-2, where 1.0 = 100%)
	float   mSpawnRotationCos; // Spawn area rotation cos
	float   mSpawnRotationSin; // Spawn area rotation sin
	int     mShowSpawnArea;    // Show spawn area preview
	float   mSpawnAreaColorR;  // Spawn area color R (pre-converted)
	float   mSpawnAreaColorG;  // Spawn area color G (pre-converted)
	float   mSpawnAreaColorB;  // Spawn area color B (pre-converted)
	float   mAlphaBoundsMinX;  // Alpha bounds min X
	float   mAlphaBoundsMinY;  // Alpha bounds min Y
	float   mAlphaBoundsWidth; // Alpha bounds width
	float   mAlphaBoundsHeight;// Alpha bounds height
	float   mOriginOffsetX;    // User origin offset X (px)
	float   mOriginOffsetY;    // User origin offset Y (px)
	int     mLineDownsample;
	int     mTileCountX;
	int     mTileCountY;
	int     mTileSize;
	float   mFrameIndex;
	float   mSeqTimeHash; // Used to invalidate cache when clip position changes
	// Focus (Depth of Field)
	int     mFocusEnable;
	float   mFocusDepth;
	float   mFocusRange;
	float   mFocusBlurStrength;
	// Motion Blur
	int     mMotionBlurEnable;
	int     mMotionBlurSamples;
	float   mMotionBlurStrength;
	// Note: Color is now stored per-line in lineData (3 Float4s per line)
};

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
	t = saturate(t);
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

float4 LoadFloat4(ByteAddressBuffer buffer, uint index)
{
	return asfloat(buffer.Load4(index * 16));
}

int LoadInt(ByteAddressBuffer buffer, uint index)
{
	return (int)buffer.Load(index * 4);
}

// Thread-block size for execution
[numthreads(16, 16, 1)]

// Root Signature determines the order in which the different elements (CBV/UAV/SRV) are expected from the host code
// We recommend using Descriptor tables over Root Descriptors
[RootSignature("DescriptorTable(CBV(b0), visibility=SHADER_VISIBILITY_ALL),DescriptorTable(UAV(u0), visibility=SHADER_VISIBILITY_ALL),DescriptorTable(SRV(t0, numDescriptors=4), visibility=SHADER_VISIBILITY_ALL)")]
void main(uint3 inXY : SV_DispatchThreadID)
{
	if (inXY.x < mWidth && inXY.y < mHeight)
	{
		uint dataSize;
		float4 pixel;
		if (mIs16f)
		{
			dataSize = sizeof(half4);
			pixel = float4(mIOImage.Load<half4>(mPitch * inXY.y + dataSize * inXY.x));
		}
		else
		{
			dataSize = sizeof(float4);
			pixel = mIOImage.Load<float4>(mPitch * inXY.y + dataSize * inXY.x);
		}
		
		// Hide original element - start with transparent black
		if (mHideElement != 0)
		{
			pixel = float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
		
		const float originalAlpha = pixel.w;  // Save original alpha for blend modes
		// Accumulate front lines separately for blend mode 2
		float3 frontColor = float3(0.0f, 0.0f, 0.0f);
		float frontAlpha = 0.0f;
		const float aa = mLineAA > 0.0f ? mLineAA : 1.0f;
		const int tileX = mTileSize > 0 ? (int)(inXY.x / (uint)mTileSize) : 0;
		const int tileY = mTileSize > 0 ? (int)(inXY.y / (uint)mTileSize) : 0;
		const int tileIndex = tileY * mTileCountX + tileX;
		const int start = LoadInt(mTileOffsets, tileIndex);
		const int count = LoadInt(mTileCounts, tileIndex);
		
		// Focus (Depth of Field) disabled
		float focusAlpha = 1.0f;
		
		// Main line pass (with shadow drawn first for each line)
		for (int j = 0; j < count; ++j)
		{
			const int lineIndex = LoadInt(mLineIndices, start + j);
			const int base = lineIndex * 3;  // 3 Float4s per line: d0, d1, d2
			const float4 d0 = LoadFloat4(mLineData, base);
			const float4 d1 = LoadFloat4(mLineData, base + 1);
			const float4 d2 = LoadFloat4(mLineData, base + 2);  // Color data
			const float lineDepthValue = d1.w;  // Stored depth value
			const float depthScale = DepthScale(lineDepthValue, mLineDepthStrength);
			const float fadeStart = 0.6f;
			const float fadeEnd = 0.2f;
			const float t = saturate((depthScale - fadeEnd) / (fadeStart - fadeEnd));
			const float depthAlpha = lerp(0.05f, 1.0f, t);

			const float centerX = d0.x;
			const float centerY = d0.y;
			const float lineCos = d0.z;
			const float lineSin = d0.w;
			
			// Pixel distance from line center
			const float dx = (float)inXY.x + 0.5f - centerX;
			const float dy = (float)inXY.y + 0.5f - centerY;
			
			// Draw shadow first (before the line)
			if (mShadowEnable != 0)
			{
				const float halfLen = d1.x;
				const float halfThick = d1.y;
				const float segCenterX = d1.z;
				
				const float scenterX = centerX + mShadowOffsetX;
				const float scenterY = centerY + mShadowOffsetY;
				
				const float sdx = (float)inXY.x + 0.5f - scenterX;
				const float sdy = (float)inXY.y + 0.5f - scenterY;
				float spx = sdx * lineCos + sdy * lineSin;
				float spy = -sdx * lineSin + sdy * lineCos;
				spx -= segCenterX;
				
				float sdist = 0.0f;
				if (mLineCap == 1)
				{
					const float ax = abs(spx) - halfLen;
					const float qx = ax > 0.0f ? ax : 0.0f;
					sdist = sqrt(qx * qx + spy * spy) - halfThick;
				}
				else
				{
					const float dxBox = abs(spx) - halfLen;
					const float dyBox = abs(spy) - halfThick;
					const float ox = dxBox > 0.0f ? dxBox : 0.0f;
					const float oy = dyBox > 0.0f ? dyBox : 0.0f;
					const float outside = sqrt(ox * ox + oy * oy);
					const float inside = min(max(dxBox, dyBox), 0.0f);
					sdist = outside + inside;
				}
				
				const float sdenom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
				const float stailT = saturate((spx + halfLen) / sdenom);
				const float stailFade = 1.0f + (stailT - 1.0f) * mLineTailFade;
				const float scoverage = smoothstep(aa, 0.0f, sdist) * stailFade * mShadowOpacity * depthAlpha;
				if (scoverage > 0.0f)
				{
					float shadowBlend = scoverage;
					if (mBlendMode == 0)
					{
						// Back: keep shadow behind the original element
						shadowBlend = scoverage * (1.0f - originalAlpha);
					}
					else if (mBlendMode == 2 && lineDepthValue < 0.5f)
					{
						// Back portion of "Back and Front": keep shadow behind the original element
						shadowBlend = scoverage * (1.0f - originalAlpha);
					}

					// Shadow: blend toward pre-converted shadow color
					// mShadowColorR/G/B are already in output format (VUYA or BGRA)
					pixel.x = pixel.x + (mShadowColorR - pixel.x) * shadowBlend;
					pixel.y = pixel.y + (mShadowColorG - pixel.y) * shadowBlend;
					pixel.z = pixel.z + (mShadowColorB - pixel.z) * shadowBlend;
					pixel.w = max(pixel.w, shadowBlend);
				}
			}
			const float halfLen = d1.x;
			float halfThick = d1.y;
			const float segCenterX = d1.z;
			// d2 contains line color (already in output color space)
			const float3 lineColor = float3(d2.x, d2.y, d2.z);
		const float lineVelocity = d2.w;  // Instantaneous velocity from easing (0-2 range typically)
			
			// Motion blur: sample multiple positions along the movement direction
			float coverage = 0.0f;
			if (mMotionBlurEnable != 0 && mMotionBlurSamples > 1)
			{
				// Motion blur enabled: sample multiple positions
				const int samples = mMotionBlurSamples;
				// Shutter angle based motion blur calculation
				const float shutterFraction = mMotionBlurStrength / 360.0f;
				const float pixelsPerFrame = mLineTravel / mLineLifetime;
				const float effectiveVelocity = pixelsPerFrame * lineVelocity;
				const float blurRange = effectiveVelocity * shutterFraction;
				const float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
				
				float accumA = 0.0f;
				
				for (int s = 0; s < samples; ++s)
				{
					// Trail mode: blur extends BEHIND the line only
					// t: 0 -> current position, 1 -> furthest back
					const float t = (float)s / max((float)(samples - 1), 1.0f);
					
					// t=0 -> current position, t=1 -> furthest back (oldest position)
					// Line moves in +X direction locally, so past position has smaller segCenterX
					const float sampleOffset = blurRange * t;
					
					// Transform to line-local space
					float pxSample = dx * lineCos + dy * lineSin;
					const float pySample = -dx * lineSin + dy * lineCos;
					// For trail: sample past positions (line was at smaller X values before)
					// segCenterX + sampleOffset shifts the reference point forward
					pxSample -= (segCenterX + sampleOffset);
					
					// Calculate distance for this sample
					float distSample = 0.0f;
					if (mLineCap == 1)
					{
						float ax = abs(pxSample) - halfLen;
						float qx = max(ax, 0.0f);
						distSample = sqrt(qx * qx + pySample * pySample) - halfThick;
					}
					else
					{
						float2 d = abs(float2(pxSample, pySample)) - float2(halfLen, halfThick);
						float2 dmax = max(d, float2(0.0f, 0.0f));
						float outside = length(dmax);
						float inside = min(max(d.x, d.y), 0.0f);
						distSample = outside + inside;
					}
					
					// Calculate coverage for this sample
					float tailT = saturate((pxSample + halfLen) / denom);
					float tailFade = 1.0f + (tailT - 1.0f) * mLineTailFade;
					float sampleCoverage = smoothstep(aa, 0.0f, distSample) * tailFade;
					
					// Simple averaging (standard motion blur)
					accumA += sampleCoverage;
				}
				
				// Average the coverage
				coverage = (accumA / max((float)samples, 1.0f)) * focusAlpha * depthAlpha;
			}
			else
			{
				// No motion blur: original single-sample calculation
				float px = dx * lineCos + dy * lineSin;
				const float py = -dx * lineSin + dy * lineCos;
				px -= segCenterX;

				float dist = 0.0f;
				if (mLineCap == 1)
				{
					float ax = abs(px) - halfLen;
					float qx = max(ax, 0.0f);
					dist = sqrt(qx * qx + py * py) - halfThick;
				}
				else
				{
					float2 d = abs(float2(px, py)) - float2(halfLen, halfThick);
					float2 dmax = max(d, float2(0.0f, 0.0f));
					float outside = length(dmax);
					float inside = min(max(d.x, d.y), 0.0f);
					dist = outside + inside;
				}

				float denom = (2.0f * halfLen) > 0.0001f ? (2.0f * halfLen) : 0.0001f;
				float tailT = saturate((px + halfLen) / denom);
				float tailFade = 1.0f + (tailT - 1.0f) * mLineTailFade;
				coverage = smoothstep(aa, 0.0f, dist) * tailFade * focusAlpha * depthAlpha;
			}
			if (coverage > 0.0f)
			{
				// Apply blend mode
				if (mBlendMode == 0)  // Back (behind element)
				{
					float backBlend = coverage * (1.0f - originalAlpha);
					pixel.x = lerp(pixel.x, lineColor.x, backBlend);
					pixel.y = lerp(pixel.y, lineColor.y, backBlend);
					pixel.z = lerp(pixel.z, lineColor.z, backBlend);
					pixel.w = max(pixel.w, backBlend);
				}
				else if (mBlendMode == 1)  // Front (in front of element)
				{
					pixel.x = lerp(pixel.x, lineColor.x, coverage);
					pixel.y = lerp(pixel.y, lineColor.y, coverage);
					pixel.z = lerp(pixel.z, lineColor.z, coverage);
					pixel.w = max(pixel.w, coverage);
				}
				else if (mBlendMode == 2)  // Back and Front (split by per-line depth)
				{
					// Use stored depth value from line data (consistent across frames)
					if (lineDepthValue < 0.5f)
					{
						// Back mode (full) -> apply to pixel immediately
						float backBlend = coverage * (1.0f - originalAlpha);
						pixel.x = lerp(pixel.x, lineColor.x, backBlend);
						pixel.y = lerp(pixel.y, lineColor.y, backBlend);
						pixel.z = lerp(pixel.z, lineColor.z, backBlend);
						pixel.w = max(pixel.w, backBlend);
					}
					else
					{
						// Front mode (full) -> accumulate separately, apply after loop
						float aFront = coverage;
						float3 premult = lineColor * aFront;
						frontColor = premult + frontColor * (1.0f - aFront);
						frontAlpha = aFront + frontAlpha * (1.0f - aFront);
					}
				}
				else if (mBlendMode == 3)  // Alpha (XOR transparency)
				{
					// Always draw line color
					pixel.x = lerp(pixel.x, lineColor.x, coverage);
					pixel.y = lerp(pixel.y, lineColor.y, coverage);
					pixel.z = lerp(pixel.z, lineColor.z, coverage);
					// XOR alpha only when overlapping element, otherwise normal blend
					if (originalAlpha > 0.0f)
					{
						pixel.w = saturate(originalAlpha + coverage - (originalAlpha * coverage * 2.0f));
					}
					else
					{
						pixel.w = max(pixel.w, coverage);
					}
				}
			}
		}

		// Apply front lines after back lines (blend mode 2)
		if (mBlendMode == 2 && frontAlpha > 0.0f)
		{
			pixel.xyz = frontColor + pixel.xyz * (1.0f - frontAlpha);
			pixel.w = frontAlpha + pixel.w * (1.0f - frontAlpha);
		}

		// Draw spawn area preview (filled with inverted colors)
		if (mShowSpawnArea != 0)
		{
			float spawnCenterX = mAlphaBoundsMinX + mAlphaBoundsWidth * 0.5f;
			float spawnCenterY = mAlphaBoundsMinY + mAlphaBoundsHeight * 0.5f;
			float halfW = mAlphaBoundsWidth * mLineSpawnScaleX * 0.5f;
			float halfH = mAlphaBoundsHeight * mLineSpawnScaleY * 0.5f;
			
			// Transform pixel position to rotated spawn space
			float relX = (inXY.x + 0.5f) - spawnCenterX - mOriginOffsetX;
			float relY = (inXY.y + 0.5f) - spawnCenterY - mOriginOffsetY;
			// Inverse rotate to check bounds
			float localX = relX * mSpawnRotationCos + relY * mSpawnRotationSin;
			float localY = -relX * mSpawnRotationSin + relY * mSpawnRotationCos;
			
			// Check if inside the spawn area (filled)
			if (abs(localX) <= halfW && abs(localY) <= halfH)
			{
				// Blend with spawn area color at 50%, then invert
				float blendAlpha = 0.5f;
				float baseX = (pixel.w <= 0.0f) ? mSpawnAreaColorR : pixel.x;
				float baseY = (pixel.w <= 0.0f) ? mSpawnAreaColorG : pixel.y;
				float baseZ = (pixel.w <= 0.0f) ? mSpawnAreaColorB : pixel.z;
				float blendedX = lerp(baseX, mSpawnAreaColorR, blendAlpha);
				float blendedY = lerp(baseY, mSpawnAreaColorG, blendAlpha);
				float blendedZ = lerp(baseZ, mSpawnAreaColorB, blendAlpha);
				pixel.x = blendedX;
				pixel.y = blendedY;
				pixel.z = blendedZ;
				pixel.w = max(pixel.w, blendAlpha);
			}
		}

		// Add tiny invisible variation to invalidate cache when clip-relative time changes
		// Apply to all pixels for stronger cache differentiation
		{
			float hashVal = frac(mSeqTimeHash * 0.0000001f);
			float cacheBuster = hashVal * 0.0000001f; // Extremely small, visually invisible
			pixel.w = saturate(pixel.w + cacheBuster);
		}

		// DEBUG: Draw BLUE TRIANGLE in top-left corner to indicate HLSL/DirectX is being used
		// Shape: Triangle (HLSL = DirectX/Windows)
		{
			const int tx = (int)inXY.x - 5;
			const int ty = (int)inXY.y - 5;
			// Triangle: y >= 0, y < 30, x >= 0, x < (30 - y) creates right triangle
			if (tx >= 0 && ty >= 0 && ty < 30 && tx < (30 - ty))
			{
				pixel.x = 0.0f;   // R = 0
				pixel.y = 0.0f;   // G = 0
				pixel.z = 1.0f;   // B = 1 (Blue for HLSL)
				pixel.w = 1.0f;   // A = 1
			}
		}
		
		if (mIs16f)
		{
			mIOImage.Store<half4>(mPitch * inXY.y + dataSize * inXY.x, half4(pixel));
		}
		else
		{
			mIOImage.Store<float4>(mPitch * inXY.y + dataSize * inXY.x, pixel);
		}
	}
}