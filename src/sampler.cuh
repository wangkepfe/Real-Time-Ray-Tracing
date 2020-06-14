#pragma once

#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include "fp16Utils.cuh"

__forceinline__ __device__ Float4 Load2D(
	SurfObj tex,
	Int2    uv)
{
    float4 ret = surf2Dread<float4>(tex, uv.x * 4 * sizeof(float), uv.y, cudaBoundaryModeClamp);
	return Float4(ret.x, ret.y, ret.z, ret.w);
}

__forceinline__ __device__ void Store2D(
    Float4  val,
	SurfObj tex,
	Int2    uv)
{
    surf2Dwrite(make_float4(val.x, val.y, val.z, val.w), tex, uv.x * 4 * sizeof(float), uv.y, cudaBoundaryModeClamp);
}

__forceinline__ __device__ Float4 Load2Dfp16(
	SurfObj tex,
	Int2    uv)
{
	ushort4 ret = surf2Dread<ushort4>(tex, uv.x * 4 * sizeof(unsigned short), uv.y, cudaBoundaryModeClamp);

	ushort4ToHalf4Converter conv(ret);
	Half4 hf4 = conv.hf4;

	return half4ToFloat4(hf4);
}

__forceinline__ __device__ void Store2Dfp16(
    Float4  fl4,
	SurfObj tex,
	Int2    uv)
{
    Half4 hf4 = float4ToHalf4(fl4);

    ushort4ToHalf4Converter conv(hf4);
    ushort4 us4 = conv.us4;

    surf2Dwrite(us4, tex, uv.x * 4 * sizeof(unsigned short), uv.y, cudaBoundaryModeClamp);
}

__device__ Float4 SampleBicubicSmoothStep(
    SurfObj tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV         = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc         = floor( UV - 0.5f ) + 0.5f;
    Float2 f          = UV - tc;

	Float2 f2 = f * f;
	Float2 f3 = f2 * f;

    Float2 w1 = -2.0f * f3 + 3.0f * f2;
    Float2 w0 = 1.0f - w1;

    Int2 tc0 = floori(UV - 0.5f);
    Int2 tc1 = tc0 + 1;

    Int2 sampleUV[4] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y },
    };

    float weights[4] = {
        w0.x * w0.y,  w1.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,
    };

	Float4 OutColor;
    float sumWeight = 0;

	#pragma unroll
	for (int i = 0; i < 4; i++)
	{
        sumWeight += weights[i];
		OutColor += Load2D(tex, sampleUV[i]) * weights[i];
	}

	OutColor /= sumWeight;

    return OutColor;
}

__device__ Float4 SampleBicubicCatmullRom(
    SurfObj tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV         = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc         = floor( UV - 0.5f ) + 0.5f;
    Float2 f          = UV - tc;

	Float2 f2 = f * f;
	Float2 f3 = f2 * f;

    Float2 w0 = f2 - 0.5f * (f3 + f);
	Float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
	Float2 w3 = 0.5f * (f3 - f2);
	Float2 w2 = 1.0f - w0 - w1 - w3;

    Int2 tc1 = floori(UV - 0.5f);
    Int2 tc0 = tc1 - 1;
    Int2 tc2 = tc1 + 1;
    Int2 tc3 = tc1 + 2;

    Int2 sampleUV[16] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y }, { tc2.x, tc0.y }, { tc3.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y }, { tc2.x, tc1.y }, { tc3.x, tc1.y },
        { tc0.x, tc2.y }, { tc1.x, tc2.y }, { tc2.x, tc2.y }, { tc3.x, tc2.y },
        { tc0.x, tc3.y }, { tc1.x, tc3.y }, { tc2.x, tc3.y }, { tc3.x, tc3.y },
    };

    float weights[16] = {
        w0.x * w0.y,  w1.x * w0.y,  w2.x * w0.y,  w3.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,  w2.x * w1.y,  w3.x * w1.y,
        w0.x * w2.y,  w1.x * w2.y,  w2.x * w2.y,  w3.x * w2.y,
        w0.x * w3.y,  w1.x * w3.y,  w2.x * w3.y,  w3.x * w3.y,
    };

	Float4 OutColor;
    float sumWeight = 0;

	#pragma unroll
	for (int i = 0; i < 16; i++)
	{
        sumWeight += weights[i];
		OutColor += Load2D(tex, sampleUV[i]) * weights[i];
	}

	OutColor /= sumWeight;

    return OutColor;
}

__device__ Float4 SampleBicubicSmoothStepfp16(
    SurfObj tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV         = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc         = floor( UV - 0.5f ) + 0.5f;
    Float2 f          = UV - tc;

	Float2 f2 = f * f;
	Float2 f3 = f2 * f;

    Float2 w1 = -2.0f * f3 + 3.0f * f2;
    Float2 w0 = 1.0f - w1;

    Int2 tc0 = floori(UV - 0.5f);
    Int2 tc1 = tc0 + 1;

    Int2 sampleUV[4] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y },
    };

    float weights[4] = {
        w0.x * w0.y,  w1.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,
    };

	Float4 OutColor;
    float sumWeight = 0;

	#pragma unroll
	for (int i = 0; i < 4; i++)
	{
        sumWeight += weights[i];
		OutColor += Load2Dfp16(tex, sampleUV[i]) * weights[i];
	}

	OutColor /= sumWeight;

    return OutColor;
}

__device__ Float4 SampleBicubicCatmullRomfp16(
    SurfObj tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV         = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc         = floor( UV - 0.5f ) + 0.5f;
    Float2 f          = UV - tc;

	Float2 f2 = f * f;
	Float2 f3 = f2 * f;

    Float2 w0 = f2 - 0.5f * (f3 + f);
	Float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
	Float2 w3 = 0.5f * (f3 - f2);
	Float2 w2 = 1.0f - w0 - w1 - w3;

    Int2 tc1 = floori(UV - 0.5f);
    Int2 tc0 = tc1 - 1;
    Int2 tc2 = tc1 + 1;
    Int2 tc3 = tc1 + 2;

    Int2 sampleUV[16] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y }, { tc2.x, tc0.y }, { tc3.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y }, { tc2.x, tc1.y }, { tc3.x, tc1.y },
        { tc0.x, tc2.y }, { tc1.x, tc2.y }, { tc2.x, tc2.y }, { tc3.x, tc2.y },
        { tc0.x, tc3.y }, { tc1.x, tc3.y }, { tc2.x, tc3.y }, { tc3.x, tc3.y },
    };

    float weights[16] = {
        w0.x * w0.y,  w1.x * w0.y,  w2.x * w0.y,  w3.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,  w2.x * w1.y,  w3.x * w1.y,
        w0.x * w2.y,  w1.x * w2.y,  w2.x * w2.y,  w3.x * w2.y,
        w0.x * w3.y,  w1.x * w3.y,  w2.x * w3.y,  w3.x * w3.y,
    };

	Float4 OutColor;
    float sumWeight = 0;

	#pragma unroll
	for (int i = 0; i < 16; i++)
	{
        sumWeight += weights[i];
		OutColor += Load2Dfp16(tex, sampleUV[i]) * weights[i];
	}

	OutColor /= sumWeight;

    return OutColor;
}


union UintFloatConverter { uint ui; float f; __device__ UintFloatConverter() : ui(0) {} };

// [x sign] [x] [y sign] [y] [z sign] [z]
//    1     10     1      9     1     10

__device__ __inline__ float EncodeNormal_R11_G10_B11(Float3 normal)
{
    const uint max9 = (0x1 << 9) - 1;
    const uint max10 = (0x1 << 10) - 1;

    UintFloatConverter converter;
    converter.ui =
		((normal.x < 0) ? (0x1 << 31) : 0) | ((uint)(fabsf(normal.x) * max10) << 21) |
		((normal.y < 0) ? (0x1 << 20) : 0) | ((uint)(fabsf(normal.y) * max9) << 11)  |
		((normal.z < 0) ? (0x1 << 10) : 0) | ((uint)(fabsf(normal.z) * max10));
    return converter.f;
}

__device__ __inline__ Float3 DecodeNormal_R11_G10_B11(float fcode)
{
	const uint max9 = (0x1 << 9) - 1;
	const uint max10 = (0x1 << 10) - 1;

    const uint maskX = 0x7fe00000; // 0111 1111 1110 0000 0000 0000 0000 0000
    const uint maskY = 0x000ff800; // 0000 0000 0000 1111 1111 1000 0000 0000
    const uint maskZ = 0x000003ff; // 0000 0000 0000 0000 0000 0011 1111 1111

    UintFloatConverter converter;
    converter.f = fcode;
	uint code = converter.ui;

    return Float3(
		((code & (0x1 << 31) != 0) ? (-1) : 1) * (float)((code & maskX) >> 21) / (float)max10,
		((code & (0x1 << 20) != 0) ? (-1) : 1) * (float)((code & maskY) >> 11) / (float)max9,
		((code & (0x1 << 10) != 0) ? (-1) : 1) * (float)((code & maskZ)) / (float)max10);
}