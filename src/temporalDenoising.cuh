#pragma once

#include "kernel.cuh"
#include "sampler.cuh"
#include "gaussian.cuh"

#define NOISE_THRESHOLD 3

__device__ __forceinline__ Float3 RgbToYcocg(const Float3& rgb)
{
	float tmp1 = rgb.x + rgb.z;
	float tmp2 = rgb.y * 2.0f;
	return Float3(tmp1 + tmp2, (rgb.x - rgb.z) * 2.0f, tmp2 - tmp1);
}

__device__ __forceinline__ Float3 YcocgToRgb(const Float3& ycocg)
{
	float tmp = ycocg.x - ycocg.z;
	return Float3(tmp + ycocg.y, ycocg.x + ycocg.z, tmp - ycocg.y) * 0.25f;
}

template<typename T>
__inline__ __device__ void WarpReduceSum(T& v) {
	const int warpSize = 32;
	#pragma unroll
	for (uint offset = warpSize / 2; offset > 0; offset /= 2)
	{
		v += __shfl_down_sync(0xffffffff, v, offset);
	}
}

__global__ void CalculateTileNoiseLevel(
	SurfObj colorBuffer,
	SurfObj depthBuffer,
	SurfObj noiseLevelBuffer,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 4;

	int x1 = x;
	int y1 = y * 2;

	int x2 = x;
	int y2 = y * 2 + 1;

	Float3Ushort1 colorAndMask1 = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1));
	float depthValue1 = Load2DHalf1(depthBuffer, Int2(x1, y1));

	Float3 colorValue1          = colorAndMask1.xyz;

	Float3Ushort1 colorAndMask2 = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2));
	float depthValue2 = Load2DHalf1(depthBuffer, Int2(x2, y2));

	Float3 colorValue2          = colorAndMask2.xyz;

	uint background1 = depthValue1 >= RayMax;
	uint background2 = depthValue2 >= RayMax;

	float lum1 = colorValue1.getmax();
	float lum12 = lum1 * lum1;

	float lum2 = colorValue2.getmax();
	float lum22 = lum2 * lum2;

	WarpReduceSum(background1);
	WarpReduceSum(background2);
	WarpReduceSum(lum1);
	WarpReduceSum(lum12);
	WarpReduceSum(lum2);
	WarpReduceSum(lum22);

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		float notSkyRatio = 1.0f - (background1 + background2) / 64.0f;

		float lumAve = (lum1 + lum2) / 64.0f;
		float lumAveSq = lumAve * lumAve;
		float lumSqAve = (lum12 + lum22) / 64.0f;

		float blockVariance = max(1e-20f, lumSqAve - lumAveSq);

		float noiseLevel = blockVariance / max(lumAveSq, 1e-20f);
		noiseLevel *= notSkyRatio;
		noiseLevel = noiseLevel;

		Int2 gridLocation = Int2(blockIdx.x, blockIdx.y);
		Store2DHalf1(noiseLevel, noiseLevelBuffer, gridLocation);
	}
}

__global__ void TileNoiseLevel8x8to16x16(SurfObj noiseLevelBuffer, SurfObj noiseLevelBuffer16)
{
	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;
	float v1 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2, y * 2));
	float v2 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2 + 1, y * 2));
	float v3 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2, y * 2 + 1));
	float v4 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2 + 1, y * 2 + 1));
	Store2DHalf1((v1 + v2 + v3 + v4) / 4, noiseLevelBuffer16, Int2(x, y));
}

__global__ void TileNoiseLevelVisualize(SurfObj colorBuffer, SurfObj noiseLevelBuffer, Int2 size)
{
	float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
	int x1 = threadIdx.x + blockIdx.x * blockDim.x;
	int y1 = threadIdx.y + blockIdx.y * blockDim.y;
	Float3Ushort1 colorAndMask1 = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1));
	Float3 colorValue1          = colorAndMask1.xyz;
	ushort maskValue1           = colorAndMask1.w;
	if (noiseLevel > 0.0005f)
	{
		if ((threadIdx.x == 0) || (threadIdx.x == (blockDim.x - 1)) || (threadIdx.y == 0) || (threadIdx.y == (blockDim.y - 1)))
		{
			Store2DHalf3Ushort1( { Float3(1, 0, 0), maskValue1 } , colorBuffer, Int2(x1, y1));
		}
	}
}

__global__ void CopyToHistoryColorDepthBuffer(
	SurfObj colorBuffer,
	SurfObj depthBuffer,
	SurfObj accumulateBuffer,
	SurfObj depthHistoryBuffer,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;
	Int2 uv(x, y);

	surf2Dwrite(surf2Dread<ushort4>(colorBuffer, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp),
	            accumulateBuffer, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp);
	surf2Dwrite(surf2Dread<ushort1>(depthBuffer, uv.x * sizeof(ushort1), uv.y, cudaBoundaryModeClamp),
	            depthHistoryBuffer, uv.x * sizeof(ushort1), uv.y, cudaBoundaryModeClamp);
}

__global__ void CopyToHistoryColorBuffer(
	SurfObj colorBuffer,
	SurfObj accumulateBuffer,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;
	Int2 uv(x, y);

	surf2Dwrite(surf2Dread<ushort4>(colorBuffer, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp),
	            accumulateBuffer, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp);
}

__global__ void SpatialFilter5x5(
	SurfObj colorBuffer,
	SurfObj normalBuffer,
	SurfObj depthBuffer,
	SurfObj noiseLevelBuffer16,
	DenoisingParams params,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 16;
	int y = threadIdx.y + blockIdx.y * 16;

	float noiseLevel = Load2DHalf1(noiseLevelBuffer16, Int2(blockIdx.x, blockIdx.y));

	if (noiseLevel < NOISE_THRESHOLD)
	{
		return;
	}

	struct AtrousLDS
	{
		Half3 color;
		ushort mask;
		Half3 normal;
		half depth;
	};
	__shared__ AtrousLDS sharedBuffer[20 * 20];

	// calculate address
	int id = (threadIdx.x + threadIdx.y * 16);
	int x1 = blockIdx.x * 16 - 2 + id % 20;
	int y1 = blockIdx.y * 16 - 2 + id / 20;
	int x2 = blockIdx.x * 16 - 2 + (id + 256) % 20;
	int y2 = blockIdx.y * 16 - 2 + (id + 256) / 20;

	// global load 1
	Half3Ushort1 colorAndMask1 = Load2DHalf3Ushort1<Half3Ushort1>(colorBuffer, Int2(x1, y1));
	sharedBuffer[id] =
	{
		colorAndMask1.xyz,
		colorAndMask1.w,
		Load2DHalf4<Half3>(normalBuffer, Int2(x1, y1)),
		Load2DHalf1<half>(depthBuffer, Int2(x1, y1))
	};

	// global load 2
	if (id + 256 < 400)
	{
		Half3Ushort1 colorAndMask2 = Load2DHalf3Ushort1<Half3Ushort1>(colorBuffer, Int2(x2, y2));
		sharedBuffer[id + 256] =
		{
			colorAndMask2.xyz,
			colorAndMask2.w,
			Load2DHalf4<Half3>(normalBuffer, Int2(x2, y2)),
			Load2DHalf1<half>(depthBuffer, Int2(x2, y2))
		};
	}

	__syncthreads();

	if (x >= size.x && y >= size.y) return;

	// load center
	AtrousLDS center   = sharedBuffer[threadIdx.x + 2 + (threadIdx.y + 2) * 20];
	Float3 colorValue  = half3ToFloat3(center.color);
	float depthValue   = __half2float(center.depth);
	Float3 normalValue = half3ToFloat3(center.normal);
	ushort maskValue   = center.mask;

	if (depthValue >= RayMax) return;

	// -------------------------------- atrous filter --------------------------------
	Float3 sumOfColor = 0;
	float sumOfWeight = 0;

	#pragma unroll
	for (int i = 0; i < 25; i += 1)
	{
		int xoffset = i % 5;
		int yoffset = i / 5;

		AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * 20];

		// get data
		Float3 color  = half3ToFloat3(bufferReadTmp.color);
		float depth   = __half2float(bufferReadTmp.depth);
		Float3 normal = half3ToFloat3(bufferReadTmp.normal);
		ushort mask   = bufferReadTmp.mask;

		float weight = 1.0f;

		// normal diff factor
		weight      *= powf(max(dot(normalValue, normal), 0.0f), params.sigma_normal);

		// depth diff fatcor
		float deltaDepth = (depthValue - depth) / params.sigma_depth;
		weight      *= expf(-0.5f * deltaDepth * deltaDepth);

		// material mask diff factor
		weight      *= (maskValue != mask) ? 1.0f / params.sigma_material : 1.0f;

		// gaussian filter weight
		weight      *= GetGaussian5x5(xoffset + yoffset * 5);

		// accumulate
		sumOfColor  += color * weight;
		sumOfWeight += weight;
	}

	Float3 finalColor;

	// final color
	if (sumOfWeight == 0)
    {
        finalColor = 0;
    }
    else
    {
        finalColor = sumOfColor / sumOfWeight;
    }

	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        printf("SpatialFilter5x5: nan found at (%d, %d)\n", x, y);
        finalColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { finalColor, maskValue } , colorBuffer, Int2(x, y));
}

#if 0
#define SAMPLE_KERNEL_SAMPLE_PER_FRAME 13
#define SAMPLE_KERNEL_FRAME_COUNT 7
#define SAMPLE_KERNEL_SIZE SAMPLE_KERNEL_SAMPLE_PER_FRAME * SAMPLE_KERNEL_FRAME_COUNT

__constant__ int SampleKernel3d[SAMPLE_KERNEL_SIZE][2] =
{{0, 0}, {5, 3}, {3, 2}, {1, 4}, {5, 6}, {3, 2}, {0, 1}, {4, 3}, {2, 5}, {4, 1}, {6, 0}, {0, 2}, {3, 4},
 {2, 6}, {1, 5}, {6, 1}, {4, 0}, {3, 5}, {6, 4}, {2, 0}, {0, 6}, {1, 4}, {1, 3}, {4, 6}, {5, 5}, {6, 3},
 {3, 1}, {2, 3}, {5, 5}, {0, 3}, {1, 0}, {4, 2}, {3, 4}, {5, 2}, {6, 6}, {0, 1}, {3, 1}, {2, 5}, {1, 0},
 {0, 4}, {6, 0}, {4, 6}, {2, 1}, {6, 3}, {1, 6}, {0, 5}, {3, 0}, {2, 2}, {5, 4}, {3, 3}, {1, 6}, {4, 1},
 {1, 2}, {5, 4}, {0, 2}, {4, 4}, {5, 1}, {5, 1}, {1, 3}, {0, 0}, {5, 5}, {2, 0}, {6, 2}, {5, 6}, {0, 4},
 {4, 5}, {3, 0}, {1, 6}, {6, 2}, {0, 4}, {2, 3}, {4, 6}, {6, 5}, {6, 3}, {3, 2}, {0, 5}, {2, 4}, {2, 2},
 {0, 1}, {3, 3}, {2, 5}, {5, 0}, {1, 2}, {2, 1}, {4, 4}, {3, 6}, {6, 1}, {4, 0}, {0, 3}, {4, 5}, {1, 1}};
#endif

__global__ void SpatialFilter7x7(
	ConstBuffer            cbo,
	SurfObj colorBuffer,
	SurfObj normalBuffer,
	SurfObj depthBuffer,
	SurfObj noiseLevelBuffer16,
	DenoisingParams params,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 16;
	int y = threadIdx.y + blockIdx.y * 16;

	float noiseLevel = Load2DHalf1(noiseLevelBuffer16, Int2(blockIdx.x, blockIdx.y));
	if (noiseLevel < 0.003f)
	{
		return;
	}

	struct AtrousLDS
	{
		Half3 color;
		ushort mask;
		Half3 normal;
		half depth;
	};

	constexpr int blockdim                  = 16;
	constexpr int kernelRadius              = 3;

	constexpr int threadCount     = blockdim * blockdim;

	constexpr int kernelCoverDim  = blockdim + kernelRadius * 2;
	constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;

	constexpr int kernelDim       = kernelRadius * 2 + 1;
	constexpr int kernelSize      = kernelDim * kernelDim;

	int centerIdx                 = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

	__shared__ AtrousLDS sharedBuffer[kernelCoverSize];

	// calculate address
	int id = (threadIdx.x + threadIdx.y * blockdim);
	int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
	int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
	int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
	int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;

	// global load 1
	Half3Ushort1 colorAndMask1 = Load2DHalf3Ushort1<Half3Ushort1>(colorBuffer, Int2(x1, y1));
	Half3Ushort1 colorAndMask2 = Load2DHalf3Ushort1<Half3Ushort1>(colorBuffer, Int2(x2, y2));
	Float3 color1 = half3ToFloat3(colorAndMask1.xyz);
	Float3 color2 = half3ToFloat3(colorAndMask2.xyz);
	if (isnan(color1.x) || isnan(color1.y) || isnan(color1.z) || isnan(color2.x) || isnan(color2.y) || isnan(color2.z))
	{
		return;
	}

	sharedBuffer[id] =
	{
		colorAndMask1.xyz,
		colorAndMask1.w,
		Load2DHalf4<Half3>(normalBuffer, Int2(x1, y1)),
		Load2DHalf1<half>(depthBuffer, Int2(x1, y1))
	};

	// global load 2
	if (id + threadCount < kernelCoverSize)
	{
		sharedBuffer[id + threadCount] =
		{
			colorAndMask2.xyz,
			colorAndMask2.w,
			Load2DHalf4<Half3>(normalBuffer, Int2(x2, y2)),
			Load2DHalf1<half>(depthBuffer, Int2(x2, y2))
		};
	}

	__syncthreads();

	if (x >= size.x && y >= size.y) return;

	// load center
	AtrousLDS center   = sharedBuffer[centerIdx];
	Float3 colorValue  = half3ToFloat3(center.color);
	float depthValue   = __half2float(center.depth);
	Float3 normalValue = half3ToFloat3(center.normal);
	ushort maskValue   = center.mask;

	NAN_DETECTER(colorValue);
	NAN_DETECTER(depthValue);
	NAN_DETECTER(normalValue);

	if (depthValue >= RayMax) return;

	// -------------------------------- atrous filter --------------------------------
	Float3 sumOfColor = 0;
	float sumOfWeight = 0;

#if 0
	#pragma unroll
	for (int i = 0; i < SAMPLE_KERNEL_SAMPLE_PER_FRAME; i += 1)
	{
		int kernelSelect = cbo.frameNum % SAMPLE_KERNEL_FRAME_COUNT;
		int kernelSelectIdx = kernelSelect * SAMPLE_KERNEL_SAMPLE_PER_FRAME + i;

		int xoffset = SampleKernel3d[kernelSelectIdx][0];
		int yoffset = SampleKernel3d[kernelSelectIdx][1];
#elif 0
	#pragma unroll
	for (int i = 0; i < kernelSize; i += 1)
	{
		int xoffset = i % kernelDim;
		int yoffset = i / kernelDim;
#elif 1
	const int stride = 2;
	int j = cbo.frameNum % stride;
	#pragma unroll
	for (int i = 0; i < kernelSize / stride; ++i)
	{
		int xoffset = j % kernelDim;
		int yoffset = j / kernelDim;
		j += stride;
#endif
		AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];

		// get data
		Float3 color  = half3ToFloat3(bufferReadTmp.color);
		float depth   = __half2float(bufferReadTmp.depth);
		Float3 normal = half3ToFloat3(bufferReadTmp.normal);
		ushort mask   = bufferReadTmp.mask;

		NAN_DETECTER(color);
		NAN_DETECTER(depth);
		NAN_DETECTER(normal);

		float weight = 1.0f;

		// normal diff factor
		weight      *= powf(max(dot(normalValue, normal), 0.0001f), params.sigma_normal);

		// depth diff fatcor
		float deltaDepth = (depthValue - depth) / params.sigma_depth;
		weight      *= expf(-0.5f * deltaDepth * deltaDepth);

		// material mask diff factor
		weight      *= (maskValue != mask) ? 1.0f / params.sigma_material : 1.0f;

		// gaussian filter weight
		weight      *= GetGaussian7x7(xoffset + yoffset * kernelDim);

		// accumulate
		sumOfColor  += color * weight;
		sumOfWeight += weight;
	}

	Float3 finalColor;

	NAN_DETECTER(sumOfColor);
	NAN_DETECTER(sumOfWeight);

	// final color
	if (sumOfWeight == 0)
    {
        finalColor = 0;
    }
    else
    {
        finalColor = sumOfColor / sumOfWeight;
    }

	NAN_DETECTER(finalColor);

	// store to current
	Store2DHalf3Ushort1( { finalColor, maskValue } , colorBuffer, Int2(x, y));
}

template<int KernelStride>
__global__ void SpatialFilterGlobal5x5(
	ConstBuffer            cbo,
	SurfObj colorBuffer,
	SurfObj normalBuffer,
	SurfObj depthBuffer,
	SurfObj noiseLevelBuffer,
	DenoisingParams params,
	Int2    size)
{
	float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
	if (noiseLevel < 0.0001f)
	{
		return;
	}

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= size.x && y >= size.y) return;

	Float3Ushort1 colorAndMaskCenter = Load2DHalf3Ushort1(colorBuffer, Int2(x, y));
	Float3 normalValue = Load2DHalf4(normalBuffer, Int2(x, y)).xyz;
	Float3 colorValue  = colorAndMaskCenter.xyz;
	ushort maskValue   = colorAndMaskCenter.w;
	float depthValue   = Load2DHalf1(depthBuffer, Int2(x, y));

	if (isnan(colorValue.x) || isnan(colorValue.y) || isnan(colorValue.z))
    {
		colorValue = 0;
    }

	NAN_DETECTER(colorValue);
	NAN_DETECTER(depthValue);
	NAN_DETECTER(normalValue);

	if (depthValue >= 10e9f) return;

	Float3 sumOfColor = 0;
	float sumOfWeight = 0;

	const int stride = 1;
	int k;
	if (stride > 1)
	{
		k = cbo.frameNum % stride;
	}
	else
	{
		k = 0;
	}

	#pragma unroll
	for (int loopIdx = 0; loopIdx < 25 / stride; ++loopIdx)
	{
		int i = k % 5;
		int j = k / 5;

		k += stride;

		Int2 loadIdx(x + (i - 2) * KernelStride, y + (j - 2) * KernelStride);

		Float3Ushort1 colorAndMask = Load2DHalf3Ushort1(colorBuffer, loadIdx);
		Float3 color  = colorAndMask.xyz;
		float depth   = Load2DHalf1(depthBuffer, loadIdx);
		Float3 normal = Load2DHalf4(normalBuffer, loadIdx).xyz;
		ushort mask   = colorAndMask.w;

		float weight = 1.0f;

		// normal diff factor
		weight      *= powf(max(dot(normalValue, normal), 0.0f), params.sigma_normal);

		// depth diff fatcor
		float deltaDepth = (depthValue - depth) / params.sigma_depth;
		weight      *= expf(-0.5f * deltaDepth * deltaDepth);

		// material mask diff factor
		weight      *= (maskValue != mask) ? 1.0f / params.sigma_material : 1.0f;

		// gaussian filter weight
		weight      *= GetGaussian5x5(i + j * 5);

		if (isnan(color.x) || isnan(color.y) || isnan(color.z))
		{
			color = 0;
			weight = 0;
		}

		NAN_DETECTER(color);

		// accumulate
		sumOfColor  += color * weight;
		sumOfWeight += weight;
	}

	NAN_DETECTER(sumOfColor);
	NAN_DETECTER(sumOfWeight);

	// final color
	Float3 finalColor;
	if (sumOfWeight == 0)
    {
        finalColor = 0;
    }
    else
    {
        finalColor = sumOfColor / sumOfWeight;
    }

	NAN_DETECTER(finalColor);

	// store to current
	Store2DHalf3Ushort1( { finalColor, maskValue } , colorBuffer, Int2(x, y));
}

__global__ void TemporalFilter(
	ConstBuffer            cbo,
	SurfObj   colorBuffer,
	SurfObj   accumulateBuffer,
	SurfObj   normalBuffer,
	SurfObj   depthBuffer,
	SurfObj   depthHistoryBuffer,
    SurfObj   motionVectorBuffer,
	SurfObj   noiseLevelBuffer,
	DenoisingParams params,
	Int2      size,
	Int2      historySize)
{
	// float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
	// if (noiseLevel < 0.003f)
	// {
	// 	return;
	// }

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	struct AtrousLDS
	{
		Half3 color;
		ushort mask;
		Half3 normal;
		half depth;
	};

	constexpr int blockdim                  = 8;
	constexpr int kernelRadius              = 1;

	constexpr int threadCount     = blockdim * blockdim;

	constexpr int kernelCoverDim  = blockdim + kernelRadius * 2;
	constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;

	constexpr int kernelDim       = kernelRadius * 2 + 1;
	constexpr int kernelSize      = kernelDim * kernelDim;

	int centerIdx                 = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

	__shared__ AtrousLDS sharedBuffer[kernelCoverSize];

	// calculate address
	int id = (threadIdx.x + threadIdx.y * blockdim);
	int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
	int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
	int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
	int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;

	Half3Ushort1 colorAndMask1 = Load2DHalf3Ushort1<Half3Ushort1>(colorBuffer, Int2(x1, y1));
	Half3Ushort1 colorAndMask2 = Load2DHalf3Ushort1<Half3Ushort1>(colorBuffer, Int2(x2, y2));
	Float3 color1 = half3ToFloat3(colorAndMask1.xyz);
	Float3 color2 = half3ToFloat3(colorAndMask2.xyz);
	if (isnan(color1.x) || isnan(color1.y) || isnan(color1.z) || isnan(color2.x) || isnan(color2.y) || isnan(color2.z))
	{
		return;
	}

	// global load 1
	sharedBuffer[id] =
	{
		colorAndMask1.xyz,
		colorAndMask1.w,
		Load2DHalf4<Half3>(normalBuffer, Int2(x1, y1)),
		Load2DHalf1<half>(depthBuffer, Int2(x1, y1))
	};

	// global load 2
	if (id + threadCount < kernelCoverSize)
	{
		sharedBuffer[id + threadCount] =
		{
			colorAndMask2.xyz,
			colorAndMask2.w,
			Load2DHalf4<Half3>(normalBuffer, Int2(x2, y2)),
			Load2DHalf1<half>(depthBuffer, Int2(x2, y2))
		};
	}

	__syncthreads();

	if (x >= size.x && y >= size.y) return;

	// load center
	AtrousLDS center   = sharedBuffer[centerIdx];
	Float3 colorValue  = half3ToFloat3(center.color);
	float depthValue   = __half2float(center.depth);
	Float3 normalValue = half3ToFloat3(center.normal);
	ushort maskValue   = center.mask;

	NAN_DETECTER(colorValue);
	NAN_DETECTER(depthValue);
	NAN_DETECTER(normalValue);

	if (depthValue >= RayMax) return;

	Float3 neighbourMax = RgbToYcocg(colorValue);
	Float3 neighbourMin = RgbToYcocg(colorValue);

	Float3 neighbourMax2 = RgbToYcocg(colorValue);
	Float3 neighbourMin2 = RgbToYcocg(colorValue);

	Float3 filteredColor = 0;
	float weightSum = 0;

	const int stride = 1;
	const bool useSoftMax = 0;

	int j;
	if (stride > 1)
	{
		j = cbo.frameNum % stride;
	}
	else
	{
		j = 0;
	}

	#pragma unroll
	for (int i = 0; i < kernelSize / stride; ++i)
	{
		int xoffset = j % kernelDim;
		int yoffset = j / kernelDim;
		j += stride;

		AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];

		// get data
		Float3 color  = half3ToFloat3(bufferReadTmp.color);
		float depth   = __half2float(bufferReadTmp.depth);
		Float3 normal = half3ToFloat3(bufferReadTmp.normal);
		ushort mask   = bufferReadTmp.mask;

		float weight = 1.0f;

		// normal diff factor
		weight      *= powf(max(dot(normalValue, normal), 0.0f), params.sigma_normal);

		// depth diff fatcor
		float deltaDepth = (depthValue - depth) / params.sigma_depth;
		weight      *= expf(-0.5f * deltaDepth * deltaDepth);

		// material mask diff factor
		weight      *= (maskValue != mask) ? 1.0f / params.sigma_material : 1.0f;

		// gaussian filter weight
		if (kernelDim == 3)
		{
			weight *= GetGaussian3x3(xoffset + yoffset * kernelDim);
		}
		else if (kernelDim == 5)
		{
			weight *= GetGaussian5x5(xoffset + yoffset * kernelDim);
		}
		else if (kernelDim == 7)
		{
			weight *= GetGaussian7x7(xoffset + yoffset * kernelDim);
		}

		// accumulate
		filteredColor  += color * weight;
		weightSum      += weight;

		// max min
		Float3 neighbourColor = RgbToYcocg(color);

		neighbourMax = max3f(neighbourMax, neighbourColor);
		neighbourMin = min3f(neighbourMin, neighbourColor);

		if (useSoftMax && (abs(xoffset - kernelDim / 2) <= 1) && (abs(yoffset - kernelDim / 2) <= 1))
		{
			neighbourMax2 = max3f(neighbourMax2, neighbourColor);
			neighbourMin2 = min3f(neighbourMin2, neighbourColor);
		}
	}

	if (useSoftMax)
	{
		neighbourMax = (neighbourMax + neighbourMax2) / 2.0f;
		neighbourMin = (neighbourMin + neighbourMin2) / 2.0f;
	}

	if (weightSum > 0)
	{
		filteredColor /= weightSum;
	}
	else
	{
		filteredColor = 0;
	}

	NAN_DETECTER(filteredColor);

    // sample history color
	Float2 motionVec = Load2DHalf2(motionVectorBuffer, Int2(x, y)) - Float2(0.5f);
	Float2 uv = (Float2(x, y) + 0.5f) * (1.0f / Float2(size.x, size.y));
	Float2 historyUv = uv + motionVec;

	// history uv out of screen
	if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0 || historyUv.y > 1.0)
	{
		Store2DHalf3Ushort1( { filteredColor, maskValue } , colorBuffer, Int2(x, y));
		return;
	}

	// sample history
	Float3 colorHistory = SampleBicubicSmoothStep(accumulateBuffer, Load2DHalf3Ushort1Float3, historyUv, historySize);

	// clamp history
	Float3 colorHistoryYcocg = RgbToYcocg(colorHistory);
	colorHistoryYcocg = clamp3f(colorHistoryYcocg, neighbourMin, neighbourMax);
	colorHistory = YcocgToRgb(colorHistoryYcocg);

	const bool enableAntiFlickering = true;
	const bool enableBlendUsingLumaHdrFactor = true;

	float lumaHistory;
	float lumaMin;
	float lumaMax;
	float lumaCurrent;

	if (enableAntiFlickering)
	{
		lumaHistory = colorHistoryYcocg.x;
		lumaMin = neighbourMin.x;
		lumaMax = neighbourMax.x;
		lumaCurrent = RgbToYcocg(colorValue).x;
	}

	// load history material mask and depth for discard history
	float discardHistory = 0;
	Int2 historyIdx = Int2(floor(historyUv.x * historySize.x), floor(historyUv.y * historySize.y));

	#pragma unroll
	for (int i = 0; i < 4; ++i)
	{
		Int2 offset(i % 2, i / 2);

		//float depthHistory = Load2DHalf1(depthBuffer, historyIdx + offset);
		ushort maskHistory = Load2DHalf3Ushort1(accumulateBuffer, historyIdx + offset).w;

		discardHistory += (maskValue != maskHistory);
	}
	discardHistory /= 4.0f;

	colorHistory = colorHistory * (1.0f - discardHistory) + filteredColor * discardHistory;

	if (enableAntiFlickering)
	{
		lumaHistory = RgbToYcocg(colorHistory).x;
	}

	Float3 outColor;

	// base blend factor
	float blendFactor = 1.0f / 64.0f;

	NAN_DETECTER(colorValue);
	NAN_DETECTER(colorHistory);

	if (enableAntiFlickering)
	{
		// anti flickering
		blendFactor = 1.0f / 8.0f;
		blendFactor *= 0.2f + 0.8f * clampf(0.5f * min( abs(lumaHistory - lumaMin), abs(lumaHistory - lumaMax) ) / max3( lumaHistory, lumaCurrent, 1e-4f ));
	}

	if (enableBlendUsingLumaHdrFactor)
	{
		// weight with luma hdr factor
		float weightA = blendFactor * max(0.0001f, 1.0f / (lumaCurrent + 4.0f));
		float weightB = (1.0f - blendFactor) * max(0.0001f, 1.0f / (lumaHistory + 4.0f));
		float weightSum = SafeDivide(1.0f, weightA + weightB);
		weightA *= weightSum;
		weightB *= weightSum;

		outColor = colorValue * weightA + colorHistory * weightB;
	}
	else
	{
		outColor = colorValue * blendFactor + colorHistory * (1.0f - blendFactor);
	}

	NAN_DETECTER(outColor);

	// store to current
	Store2DHalf3Ushort1( { outColor, maskValue } , colorBuffer, Int2(x, y));
}


__global__ void TemporalFilter2(
	ConstBuffer            cbo,
	SurfObj   colorBuffer,
	SurfObj   accumulateBuffer,
	SurfObj   normalBuffer,
	SurfObj   depthBuffer,
	SurfObj   depthHistoryBuffer,
    SurfObj   motionVectorBuffer,
	SurfObj   noiseLevelBuffer,
	Int2      size,
	Int2      historySize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	struct AtrousLDS
	{
		Float3 color;
		int mask;
	};

	constexpr int blockdim                  = 8;
	constexpr int kernelRadius              = 1;

	constexpr int threadCount     = blockdim * blockdim;

	constexpr int kernelCoverDim  = blockdim + kernelRadius * 2;
	constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;

	constexpr int kernelDim       = kernelRadius * 2 + 1;
	constexpr int kernelSize      = kernelDim * kernelDim;

	int centerIdx                 = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

	__shared__ AtrousLDS sharedBuffer[kernelCoverSize];

	// calculate address
	int id = (threadIdx.x + threadIdx.y * blockdim);
	int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
	int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
	int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
	int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;

	// global load 1
	Float3Ushort1 colorAndMask1 = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1));
	sharedBuffer[id] =
	{
		RgbToYcocg(colorAndMask1.xyz),
		colorAndMask1.w
	};

	// global load 2
	if (id + threadCount < kernelCoverSize)
	{
		Float3Ushort1 colorAndMask2 = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2));
		sharedBuffer[id + threadCount] =
		{
			RgbToYcocg(colorAndMask2.xyz),
			colorAndMask2.w
		};
	}

	__syncthreads();

	if (x >= size.x && y >= size.y) return;

	// load center
	AtrousLDS center   = sharedBuffer[centerIdx];
	Float3 colorValue  = YcocgToRgb(center.color);
	int maskValue   = center.mask;

	Float3 neighbourMax = Float3(FLT_MIN);
	Float3 neighbourMin = Float3(FLT_MAX);

	Float3 neighbourMax2 = Float3(FLT_MIN);
	Float3 neighbourMin2 = Float3(FLT_MAX);

	const int stride = 1;
	const bool useSoftMax = 1;

	int j;
	if (stride > 1)
	{
		j = cbo.frameNum % stride;
	}
	else
	{
		j = 0;
	}

	#pragma unroll
	for (int i = 0; i < kernelSize / stride; ++i)
	{
		int xoffset = j % kernelDim;
		int yoffset = j / kernelDim;
		j += stride;

		AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];
		Float3    color         = bufferReadTmp.color;
		int       mask          = bufferReadTmp.mask;

		if (mask == maskValue)
		{
			neighbourMax = max3f(neighbourMax, color);
			neighbourMin = min3f(neighbourMin, color);

			if (useSoftMax && (abs(xoffset - kernelDim / 2) + abs(yoffset - kernelDim / 2) <= 1))
			{
				neighbourMax2 = max3f(neighbourMax2, color);
				neighbourMin2 = min3f(neighbourMin2, color);
			}
		}
	}

	if (useSoftMax)
	{
		neighbourMax = (neighbourMax + neighbourMax2) / 2.0f;
		neighbourMin = (neighbourMin + neighbourMin2) / 2.0f;
	}

    // sample history color
	Float2 motionVec = Load2DHalf2(motionVectorBuffer, Int2(x, y)) - Float2(0.5f);
	Float2 uv = (Float2(x, y) + 0.5f) * (1.0f / Float2(size.x, size.y));
	Float2 historyUv = uv + motionVec;

	// history uv out of screen
	if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0 || historyUv.y > 1.0)
	{
		return;
	}

	// sample history
	Float3 colorHistory = SampleBicubicSmoothStep(accumulateBuffer, Load2DHalf3Ushort1Float3, historyUv, historySize);

	// clamp history
	Float3 colorHistoryYcocg = RgbToYcocg(colorHistory);
	colorHistoryYcocg = clamp3f(colorHistoryYcocg, neighbourMin, neighbourMax);
	colorHistory = YcocgToRgb(colorHistoryYcocg);

	const bool enableAntiFlickering = true;
	const bool enableBlendUsingLumaHdrFactor = true;

	float lumaHistory;
	float lumaMin;
	float lumaMax;
	float lumaCurrent;

	if (enableAntiFlickering)
	{
		lumaHistory = colorHistoryYcocg.x;
		lumaMin = neighbourMin.x;
		lumaMax = neighbourMax.x;
		lumaCurrent = RgbToYcocg(colorValue).x;
	}

	// load history material mask and depth for discard history
	float discardHistory = 0;
	Int2 historyIdx = Int2(floor(historyUv.x * historySize.x), floor(historyUv.y * historySize.y));

	#pragma unroll
	for (int i = 0; i < 4; ++i)
	{
		Int2 offset(i % 2, i / 2);

		float depthHistory = Load2DHalf1(depthBuffer, historyIdx + offset);
		ushort maskHistory = Load2DHalf3Ushort1(accumulateBuffer, historyIdx + offset).w;

		discardHistory += (maskValue != maskHistory);
	}
	discardHistory /= 4.0f;

	if (discardHistory == 1.0f)
	{
		return;
	}

	colorHistory = colorHistory * (1.0f - discardHistory) + colorValue * discardHistory;

	if (enableAntiFlickering)
	{
		lumaHistory = RgbToYcocg(colorHistory).x;
	}

	Float3 outColor;

	// base blend factor
	float blendFactor = 1.0f / 64.0f;

	if (enableAntiFlickering)
	{
		// anti flickering
		blendFactor = 3.0f / 4.0f;
		blendFactor *= 0.2f + 0.8f * clampf(0.5f * min( abs(lumaHistory - lumaMin), abs(lumaHistory - lumaMax) ) / max3( lumaHistory, lumaCurrent, 1e-4f ));
	}

	if (enableBlendUsingLumaHdrFactor)
	{
		// weight with luma hdr factor
		float weightA = blendFactor * max(0.0001f, 1.0f / (lumaCurrent + 4.0f));
		float weightB = (1.0f - blendFactor) * max(0.0001f, 1.0f / (lumaHistory + 4.0f));
		float weightSum = SafeDivide(1.0f, weightA + weightB);
		weightA *= weightSum;
		weightB *= weightSum;

		outColor = colorValue * weightA + colorHistory * weightB;
	}
	else
	{
		outColor = colorValue * blendFactor + colorHistory * (1.0f - blendFactor);
	}

	NAN_DETECTER(outColor);

	// store to current
	Store2DHalf3Ushort1( { outColor, maskValue } , colorBuffer, Int2(x, y));
}

__global__ void ApplyAlbedo(SurfObj colorBuffer, SurfObj albedoBuffer, Int2 texSize)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= texSize.x || idx.y >= texSize.y) return;

    Float3 color = Load2DHalf4(colorBuffer, idx).xyz;
    Float3 albedo = Load2DHalf4(albedoBuffer, idx).xyz;

    Store2DHalf4(Float4(color * albedo, 1.0f), colorBuffer, idx);
}