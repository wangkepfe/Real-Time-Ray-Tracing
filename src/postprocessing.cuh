
#pragma once

#include <cuda_runtime.h>
#include "sampler.cuh"

#define USE_CATMULL_ROM_SAMPLER 1
#define USE_BICUBIC_SMOOTH_STEP_SAMPLER 0

__device__ Float3 Uncharted2Tonemap(Float3 x)
{
	const float A = 0.15;
	const float B = 0.50;
	const float C = 0.10;
	const float D = 0.20;
	const float E = 0.02;
	const float F = 0.30;

	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

__global__ void ToneMapping(
	SurfObj   colorBuffer,
	Int2                  size,
	float                 exposure)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;


	const float W = 11.2;

	Float3 texColor = Load2D(colorBuffer, idx).xyz;
	texColor *= exposure;

	float ExposureBias = 2.0f;
	Float3 curr = Uncharted2Tonemap(ExposureBias * texColor);

	Float3 whiteScale = 1.0f/Uncharted2Tonemap(W);
	Float3 color = curr * whiteScale;

	Float3 retColor = clamp3f(pow3f(color, 1.0f / 2.2f), Float3(0), Float3(1));

	Store2D(Float4(retColor, 1.0), colorBuffer, idx);
}

__global__ void TAA(
	SurfObj   finalBuffer,
	SurfObj   currentBuffer,
	Int2                  size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 currentColor = Load2D(currentBuffer, idx).xyz;
	Float3 historyColor = Load2D(finalBuffer, idx).xyz;

	float blendFactor = 0.3f;
	Float3 outColor = currentColor * blendFactor + historyColor * (1.0f - blendFactor);

    Store2D(Float4(outColor, 1.0), finalBuffer, idx);
}

__global__ void BufferAdd(
	SurfObj   OutBuffer,
	SurfObj   inBuffer,
	Int2                  size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 color1 = Load2D(inBuffer, idx).xyz;
	Float3 color2 = Load2D(OutBuffer, idx).xyz;

    Store2D(Float4(color1 + color2, 1.0), OutBuffer, idx);
}

__global__ void BufferDivide(
	SurfObj   OutBuffer,
	float     a,
	Int2                  size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

    Store2D(Float4(Load2D(OutBuffer, idx).xyz / a, 1.0), OutBuffer, idx);
}

__global__ void BufferCopy(
	SurfObj   OutBuffer,
	SurfObj   inBuffer,
	Int2                  size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

    Store2D(Load2D(inBuffer, idx), OutBuffer, idx);
}

__global__ void BufferInit(
	SurfObj   OutBuffer,
	Int2      size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

    Store2D(Float4(0), OutBuffer, idx);
}

__global__ void FilterScale(
	SurfObj* renderTarget,
	SurfObj  finalColorBuffer,
	Int2                 outSize,
	Int2                 texSize)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= outSize.x || idx.y >= outSize.y) return;

	Float2 uv((float)idx.x / outSize.x, (float)idx.y / outSize.y);

#if USE_BICUBIC_SMOOTH_STEP_SAMPLER
	Float3 sampledColor = SampleBicubicSmoothStep(finalColorBuffer, uv, texSize).xyz;
#elif USE_CATMULL_ROM_SAMPLER
	Float3 sampledColor = SampleBicubicCatmullRom(finalColorBuffer, uv, texSize).xyz;
#endif

	sampledColor = clamp3f(sampledColor, Float3(0), Float3(1));

	surf2Dwrite(make_uchar4(sampledColor.x * 255,
							sampledColor.y * 255,
							sampledColor.z * 255,
							1.0),
		        renderTarget[0],
				idx.x * 4,
				idx.y);
}