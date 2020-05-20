#pragma once

#include <cuda_runtime.h>
#include "sampler.cuh"
#include "debug_util.cuh"

__global__ void Denoise(
	SurfObj   colorBuffer, // [out]
	SurfObj   normalBuffer,   // [in]
	SurfObj   positionBuffer, // [in]
	Int2                  size,           // [in]
	CbDenoise             cbDenoise)      // [in]
{
	const float filterKernel[25] = {
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0,  3.0 / 32.0, 3.0 / 128.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };

	const Int2 uvOffset[25] = {
		Int2(-2,-2), Int2(-1,-2), Int2(0,-2), Int2(1,-2), Int2(2,-2),
		Int2(-2,-1), Int2(-1,-1), Int2(0,-2), Int2(1,-1), Int2(2,-1),
		Int2(-2, 0), Int2(-1, 0), Int2(0, 0), Int2(1, 0), Int2(2, 0),
		Int2(-2, 1), Int2(-1, 1), Int2(0, 1), Int2(1, 1), Int2(2, 1),
		Int2(-2, 2), Int2(-1, 2), Int2(0, 2), Int2(1, 2), Int2(2, 2) };

	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 positionValue = Load2Dfp16(positionBuffer, idx).xyz;

	if (positionValue.z > 0.02) return;

	Float3 out = Float3(0.0);
	float cum_w = 0.0;

	Float3 colorValue    = Load2D(colorBuffer, idx).xyz;
	Float3 normalValue   = Load2Dfp16(normalBuffer, idx).xyz;

	for (int i = 0; i < 25; ++i)
	{
		Int2 uv = idx + uvOffset[i];

		Float3 ctmp = Load2D(colorBuffer, uv).xyz;
        Float3 t = colorValue - ctmp;
        float dist2 = dot(t,t);
        float c_w = min1f(expf(-(dist2)/cbDenoise.c_phi), 1.0);

		Float3 ntmp = Load2D(normalBuffer, uv).xyz;
        t = normalValue - ntmp;
        dist2 = max1f(dot(t,t), 0.0);
        float n_w = min1f(expf(-(dist2)/cbDenoise.n_phi), 1.0);

        Float3 ptmp = Load2D(positionBuffer, uv).xyz;
        t = positionValue - ptmp;
        dist2 = dot(t,t);
        float p_w = min1f(expf(-(dist2)/cbDenoise.p_phi),1.0);

		float weight = c_w * n_w * p_w;
        out += ctmp * weight * filterKernel[i];
        cum_w += weight * filterKernel[i];
	}

	out /= cum_w;

	Store2D(Float4(out, 1.0), colorBuffer, idx);
}

__global__ void DenoiseV2(
	SurfObj   colorBuffer, // [in/out]
	Int2      size)        // [in]
{
	Print("123123");
	const float c_phi = 1000.0;
	const float n_phi = 1000.0;

	const float filterKernel[25] = {
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0,  3.0 / 32.0, 3.0 / 128.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };

	const Int2 uvOffset[25] = {
		Int2(-2,-2), Int2(-1,-2), Int2(0,-2), Int2(1,-2), Int2(2,-2),
		Int2(-2,-1), Int2(-1,-1), Int2(0,-2), Int2(1,-1), Int2(2,-1),
		Int2(-2, 0), Int2(-1, 0), Int2(0, 0), Int2(1, 0), Int2(2, 0),
		Int2(-2, 1), Int2(-1, 1), Int2(0, 1), Int2(1, 1), Int2(2, 1),
		Int2(-2, 2), Int2(-1, 2), Int2(0, 2), Int2(1, 2), Int2(2, 2) };

	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	//if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 out = Float3(0.0);
	float cum_w = 0.0;

	Float4 bufferRead = Load2D(colorBuffer, idx);
	Float3 colorValue = bufferRead.xyz;
	Float3 normalValue = DecodeNormal_R11_G10_B11(bufferRead.w);

	//if (normalValue.x < 1e-2 && normalValue.y < 1e-2 && normalValue.z < 1e-2)
	//{
	//	return;
	//}

	for (int i = 0; i < 25; ++i)
	{
		Int2 uv = idx + uvOffset[i];

		Float4 bufferReadTmp = Load2D(colorBuffer, uv);
		Float3 ctmp = bufferReadTmp.xyz;
		Float3 ntmp = DecodeNormal_R11_G10_B11(bufferReadTmp.w);

        Float3 t = colorValue - ctmp;
        float dist2 = dot(t,t);
        float c_w = min1f(expf(-(dist2)/c_phi), 1.0);
		c_w = 1.0;

        t = normalValue - ntmp;
        dist2 = max1f(dot(t,t), 0.0);
        float n_w = min1f(expf(-(dist2)/n_phi), 1.0);
		n_w = 1.0;

		float weight = c_w * n_w;
        out += ctmp * weight * filterKernel[i];
        cum_w += weight * filterKernel[i];
	}

	if (IsPixelAt(0.5, 0.8))
	{
		Print("colorValue", colorValue);
		Print("normalValue", normalValue);
		Print("out", out);
		Print("cum_w", cum_w);
	}

	out /= cum_w;

	Store2D(Float4(out, 1.0), colorBuffer, idx);

	// if (normalValue.x < 0)
	// {
	// 	Print("normalValue", normalValue);
	// }

	//Store2D(Float4(fabsf(normalValue.x), fabsf(normalValue.y), fabsf(normalValue.z), 1.0), colorBuffer, idx);
}