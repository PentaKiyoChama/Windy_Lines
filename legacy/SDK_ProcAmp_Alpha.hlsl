RWByteAddressBuffer gBounds : register(u0);
ByteAddressBuffer gInput : register(t0);

cbuffer cb : register(b0)
{
	int mPitch;
	int mIs16f;
	int mWidth;
	int mHeight;
	int mStride;
	float mThreshold;
};

[numthreads(16, 16, 1)]
[RootSignature("DescriptorTable(CBV(b0), visibility=SHADER_VISIBILITY_ALL),DescriptorTable(UAV(u0), visibility=SHADER_VISIBILITY_ALL),DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_ALL)")]
void main(uint3 inXY : SV_DispatchThreadID)
{
	if (inXY.x >= mWidth || inXY.y >= mHeight)
	{
		return;
	}

	if ((int)(inXY.x % (uint)mStride) != 0 || (int)(inXY.y % (uint)mStride) != 0)
	{
		return;
	}

	const uint dataSize = mIs16f ? sizeof(half4) : sizeof(float4);
	const uint index = mPitch * inXY.y + dataSize * inXY.x;
	float alpha = 0.0f;
	if (mIs16f)
	{
		alpha = (float)gInput.Load<half4>(index).w;
	}
	else
	{
		alpha = gInput.Load<float4>(index).w;
	}
	if (alpha > mThreshold)
	{
		gBounds.InterlockedMin(0, (int)inXY.x);
		gBounds.InterlockedMin(4, (int)inXY.y);
		gBounds.InterlockedMax(8, (int)inXY.x);
		gBounds.InterlockedMax(12, (int)inXY.y);
	}
}
