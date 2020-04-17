#pragma once

#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include "morton.cuh"

__device__ Float4 Load2D(
	SurfObj tex,
	Int2                uv)
{
    float4 ret = surf2Dread<float4>(tex, uv.x * 16, uv.y, cudaBoundaryModeClamp);
	return Float4(ret.x, ret.y, ret.z, ret.w);
}

__device__ void Store2D(
    Float4              val,
	SurfObj tex,
	Int2                uv)
{
    surf2Dwrite(make_float4(val.x, val.y, val.z, val.w), tex, uv.x * 16, uv.y, cudaBoundaryModeClamp);
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