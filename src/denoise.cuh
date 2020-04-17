#pragma once

#include <cuda_runtime.h>
#include "sampler.cuh"

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

	Float3 positionValue = Load2D(positionBuffer, idx).xyz;

	if (positionValue.z > 0.02) return;

	Float3 out = Float3(0.0);
	float cum_w = 0.0;

	Float3 colorValue    = Load2D(colorBuffer, idx).xyz;
	Float3 normalValue   = Load2D(normalBuffer, idx).xyz;

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