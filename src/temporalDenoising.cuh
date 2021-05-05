#pragma once

#include "kernel.cuh"
#include "sampler.cuh"
#include "common.cuh"

extern __constant__ float filterKernel[25];

__constant__ float Gaussian3x3[9] =
{
    1.0f/16.0f, 1.0f/8.0f, 1.0f / 16.0f,
	1.0f / 8.0f, 1.0f/4.0f, 1.0f / 8.0f,
	1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f,
};

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
	SurfObj normalDepthBuffer,
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
	Float2 normalAndDepth1      = Load2DFloat2(normalDepthBuffer, Int2(x1, y1));

	Float3 colorValue1          = colorAndMask1.xyz;
	float depthValue1           = normalAndDepth1.y;

	Float3Ushort1 colorAndMask2 = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2));
	Float2 normalAndDepth2      = Load2DFloat2(normalDepthBuffer, Int2(x2, y2));

	Float3 colorValue2          = colorAndMask2.xyz;
	float depthValue2           = normalAndDepth2.y;

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
		noiseLevel = min(noiseLevel * 10.0f, 1.0f);

		Int2 gridLocation = Int2(blockIdx.x, blockIdx.y);
		Store2DHalf1(noiseLevel, noiseLevelBuffer, gridLocation);
	}
}

__global__ void TileNoiseLevelVisualize(SurfObj colorBuffer, SurfObj noiseLevelBuffer, Int2 size)
{
	float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
	int x1 = threadIdx.x + blockIdx.x * 8;
	int y1 = threadIdx.y + blockIdx.y * 8;
	Float3Ushort1 colorAndMask1 = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1));
	Float3 colorValue1          = colorAndMask1.xyz;
	ushort maskValue1           = colorAndMask1.w;
	Store2DHalf3Ushort1( { colorValue1 + Float3(noiseLevel, 0, 0), maskValue1 } , colorBuffer, Int2(x1, y1));
}

__global__ void CopyToHistoryBuffer(
	SurfObj colorBuffer,
	SurfObj normalDepthBuffer,
	SurfObj accumulateBuffer,
	SurfObj normalDepthHistoryBuffer,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;
	Float3Ushort1 colorAndMask = Load2DHalf3Ushort1(colorBuffer, Int2(x, y));
	Float2 normalAndDepth      = Load2DFloat2(normalDepthBuffer, Int2(x, y));
	Store2DHalf3Ushort1( colorAndMask , accumulateBuffer, Int2(x, y));
	Store2DFloat2( normalAndDepth , normalDepthHistoryBuffer, Int2(x, y));
}

__global__ void SpatialFilterReconstruction5x5(
	SurfObj colorBuffer,
	SurfObj bsdfOverPdfBuffer,
	SurfObj noiseLevelBuffer,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 16;
	int y = threadIdx.y + blockIdx.y * 16;

	float noiseLevel1 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2, blockIdx.y * 2));
	float noiseLevel2 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2 + 1, blockIdx.y * 2));
	float noiseLevel3 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2, blockIdx.y * 2 + 1));
	float noiseLevel4 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2 + 1, blockIdx.y * 2 + 1));

	if ((noiseLevel1 < 0.1f) && (noiseLevel2 < 0.1f) && (noiseLevel3 < 0.1f) && (noiseLevel4 < 0.1f))
	{
		return;
	}

	ushort mask = Load2DHalf3Ushort1(colorBuffer, Int2(x, y)).w;

	struct AtrousLDS
	{
		Float3 color;
		Float3 brdfOverPdf;
	};
	__shared__ AtrousLDS sharedBuffer[20 * 20];

	// calculate address
	int id = (threadIdx.x + threadIdx.y * 16);

	int x1 = blockIdx.x * 16 - 2 + id % 20;
	int y1 = blockIdx.y * 16 - 2 + id / 20;

	int x2 = blockIdx.x * 16 - 2 + (id + 256) % 20;
	int y2 = blockIdx.y * 16 - 2 + (id + 256) / 20;

	// global load 1
	Float3 colorValue1  = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1)).xyz;
	Float3 bsdfOverPdf1 = Load2DHalf4(bsdfOverPdfBuffer, Int2(x1, y1)).xyz;

	// store to lds 1
	sharedBuffer[id      ] = { colorValue1, bsdfOverPdf1 };

	if (id + 256 < 400)
	{
		// global load 2
		Float3 colorValue2  = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2)).xyz;
		Float3 bsdfOverPdf2 = Load2DHalf4(bsdfOverPdfBuffer, Int2(x2, y2)).xyz;

		// store to lds 2
		sharedBuffer[id + 256] = { colorValue2, bsdfOverPdf2 };
	}

	__syncthreads();

	if (x >= size.x && y >= size.y) return;

	// load center
	AtrousLDS center   = sharedBuffer[threadIdx.x + 2 + (threadIdx.y + 2) * 20];
	Float3 colorValue  = center.color;
	Float3 brdfOverPdfValue = center.brdfOverPdf;

	if (brdfOverPdfValue == 1.0f) return;

	// -------------------------------- atrous filter --------------------------------
	Float3 sumOfColor = 0;
	Float3 sumOfWeight = 0;

	#pragma unroll
	for (int i = 0; i < 25; i += 1)
	{
		int xoffset = i % 5;
		int yoffset = i / 5;

		AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * 20];

		// get data
		Float3 color  = bufferReadTmp.color;
		Float3 brdfOverPdf = bufferReadTmp.brdfOverPdf;

		Float3 weight = brdfOverPdf;

		// accumulate
		sumOfColor  += color * weight;
		sumOfWeight += weight;
	}

	// final color
	Float3 finalColor = SafeDivide3f(sumOfColor, sumOfWeight);

	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        printf("SpatialFilterReconstruction5x5: nan found at (%d, %d)\n", x, y);
        finalColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { finalColor, mask } , colorBuffer, Int2(x, y));
}

__global__ void SpatialFilter5x5(
	SurfObj colorBuffer,
	SurfObj normalDepthBuffer,
	SurfObj noiseLevelBuffer,
	Int2    size)
{
	int x = threadIdx.x + blockIdx.x * 16;
	int y = threadIdx.y + blockIdx.y * 16;

	float noiseLevel1 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2, blockIdx.y * 2));
	float noiseLevel2 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2 + 1, blockIdx.y * 2));
	float noiseLevel3 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2, blockIdx.y * 2 + 1));
	float noiseLevel4 = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x * 2 + 1, blockIdx.y * 2 + 1));

	if ((noiseLevel1 < 0.1f) && (noiseLevel2 < 0.1f) && (noiseLevel3 < 0.1f) && (noiseLevel4 < 0.1f))
	{
		return;
	}

	struct AtrousLDS
	{
		Float3 color;
		float depth;
		Float3 normal;
		ushort mask;
	};
	__shared__ AtrousLDS sharedBuffer[20 * 20];

	// calculate address
	int id = (threadIdx.x + threadIdx.y * 16);

	int x1 = blockIdx.x * 16 - 2 + id % 20;
	int y1 = blockIdx.y * 16 - 2 + id / 20;

	int x2 = blockIdx.x * 16 - 2 + (id + 256) % 20;
	int y2 = blockIdx.y * 16 - 2 + (id + 256) / 20;

	// global load 1
	Float3Ushort1 colorAndMask1 = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1));
	Float2 normalAndDepth1      = Load2DFloat2(normalDepthBuffer, Int2(x1, y1));

	Float3 colorValue1          = colorAndMask1.xyz;
	float depthValue1           = normalAndDepth1.y;
	Float3 normalValue1         = DecodeNormal_R11_G10_B11(normalAndDepth1.x);
	ushort maskValue1           = colorAndMask1.w;

	// store to lds 1
	sharedBuffer[id      ] = { colorValue1, depthValue1, normalValue1, maskValue1 };

	if (id + 256 < 400)
	{
		// global load 2
		Float3Ushort1 colorAndMask2 = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2));
		Float2 normalAndDepth2 = Load2DFloat2(normalDepthBuffer, Int2(x2, y2));

		Float3 colorValue2 = colorAndMask2.xyz;
		float depthValue2 = normalAndDepth2.y;
		Float3 normalValue2 = DecodeNormal_R11_G10_B11(normalAndDepth2.x);
		ushort maskValue2 = colorAndMask2.w;

		// store to lds 2
		sharedBuffer[id + 256] = { colorValue2, depthValue2, normalValue2, maskValue2 };
	}

	__syncthreads();

	if (x >= size.x && y >= size.y) return;

	// load center
	AtrousLDS center   = sharedBuffer[threadIdx.x + 2 + (threadIdx.y + 2) * 20];
	Float3 colorValue  = center.color;
	float depthValue   = center.depth;
	Float3 normalValue = center.normal;
	ushort maskValue   = center.mask;

	if (depthValue >= RayMax) return;

	// -------------------------------- atrous filter --------------------------------
	const float sigma_normal = 128.0f;
	const float sigma_depth = 4.0f;
	const float sigma_material = 2.0f;

	Float3 sumOfColor = 0;
	float sumOfWeight = 0;

	#pragma unroll
	for (int i = 0; i < 25; i += 1)
	{
		int xoffset = i % 5;
		int yoffset = i / 5;

		AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * 20];

		// get data
		Float3 color  = bufferReadTmp.color;
		float depth   = bufferReadTmp.depth;
		Float3 normal = bufferReadTmp.normal;
		ushort mask   = bufferReadTmp.mask;

		float weight = 1.0f;

		// normal diff factor
		weight      *= powf(max(dot(normalValue, normal), 0.0f), sigma_normal);

		// depth diff fatcor
		float deltaDepth = (depthValue - depth) / sigma_depth;
		deltaDepth = deltaDepth * deltaDepth;
		weight      *= expf(-0.5f * deltaDepth);

		// material mask diff factor
		weight      *= (maskValue != mask) ? 1.0f / sigma_material : 1.0f;

		// gaussian filter weight
		weight      *= filterKernel[xoffset + yoffset * 5];

		// accumulate
		sumOfColor  += color * weight;
		sumOfWeight += weight;
	}

	// final color
	Float3 finalColor = SafeDivide3f1f(sumOfColor, sumOfWeight);

	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        printf("SpatialFilter5x5: nan found at (%d, %d)\n", x, y);
        finalColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { finalColor, maskValue } , colorBuffer, Int2(x, y));
}

__global__ void SpatialFilter10x10(
	SurfObj   colorBuffer,
	SurfObj   normalDepthBuffer,
	SurfObj   noiseLevelBuffer,
	Int2      size)
{
	float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
	if (noiseLevel < 0.1f)
	{
		return;
	}

	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;

	if (x >= size.x && y >= size.y) return;

	// global load
	Float3Ushort1 colorAndMask = Load2DHalf3Ushort1(colorBuffer, Int2(x, y));
	Float2 normalAndDepth      = Load2DFloat2(normalDepthBuffer, Int2(x, y));

	// get data
	Float3 colorValue  = colorAndMask.xyz;
	float depthValue   = normalAndDepth.y;
	Float3 normalValue = DecodeNormal_R11_G10_B11(normalAndDepth.x);
	ushort maskValue   = colorAndMask.w;

	Float3 sumOfColor = 0;
	float sumOfWeight = 0;
	#pragma unroll
	for (int j = 0; j <= 4; j += 1)
	{
		#pragma unroll
		for (int i = 0; i <= 4; i += 1)
		{
			// global load
			Float3Ushort1 colorAndMask = Load2DHalf3Ushort1(colorBuffer, Int2(x + i * 2 - 4, y + j * 2 - 4));
			Float2 normalAndDepth      = Load2DFloat2(normalDepthBuffer, Int2(x + i * 2 - 4, y + j * 2 - 4));

			// get data
			Float3 color  = colorAndMask.xyz;
			float depth   = normalAndDepth.y;
			Float3 normal = DecodeNormal_R11_G10_B11(normalAndDepth.x);
			ushort mask   = colorAndMask.w;

			Float3 t;
			float dist2;
			float weight = 1.0f;

			// normal diff factor
			t            = normalValue - normal;
			dist2        = dot(t,t);
			weight      *= min1f(expf(-(dist2) / 0.1f), 1.0f);

			// color diff factor
			t            = colorValue - color;
			dist2        = dot(t,t);
			weight      *= min1f(expf(-(dist2) / 0.1f), 1.0f);

			// depth diff fatcor
			dist2        = depthValue - depth;
			dist2        = dist2 * dist2;
			weight      *= min1f(expf(-(dist2) / 0.1f), 1.0f);

			// material mask diff factor
			dist2        = (maskValue != mask) ? 1.0f : 0.0f;
			weight      *= min1f(expf(-(dist2) / 0.1f), 1.0f);

			// gaussian filter weight
			weight      *= filterKernel[j + i * 5];

			// accumulate
			sumOfColor  += color * weight;
			sumOfWeight += weight;
		}
	}

	// final color
	Float3 finalColor = SafeDivide3f1f(sumOfColor, sumOfWeight);

	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        printf("SpatialFilter10x10: nan found at (%d, %d)\n", x, y);
        finalColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { finalColor, maskValue } , colorBuffer, Int2(x, y));
}

__global__ void TemporalFilter(
	SurfObj   colorBuffer,
	SurfObj   accumulateBuffer,
	SurfObj   normalDepthBuffer,
	SurfObj   normalDepthHistoryBuffer,
    SurfObj   motionVectorBuffer,
	SurfObj   noiseLevelBuffer,
	Int2      size,
	Int2      historySize)
{
	float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
	if (noiseLevel < 0.1f)
	{
		return;
	}

	__shared__ Float3Ushort1 sharedBuffer[10 * 10];

	// calculate address
	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;

	int id = (threadIdx.x + threadIdx.y * 8);

	int x1 = blockIdx.x * 8 - 1 + id % 10;
	int y1 = blockIdx.y * 8 - 1 + id / 10;

	int x2 = blockIdx.x * 8 - 1 + (id + 64) % 10;
	int y2 = blockIdx.y * 8 - 1 + (id + 64) / 10;

	// load current color and mask
	sharedBuffer[id] = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1));

	if (id + 64 < 100)
	{
		sharedBuffer[id + 64] = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2));
	}

	__syncthreads();

	if (x >= size.x || y >= size.y) return;

	// load current normal and depth
	Float2 normalAndDepth = Load2DFloat2(normalDepthBuffer, Int2(x, y));
	Float3 normal = DecodeNormal_R11_G10_B11(normalAndDepth.x);
	float depth = normalAndDepth.y;

	// current center color and mask
	Float3Ushort1 center = sharedBuffer[threadIdx.x + 1 + (threadIdx.y + 1) * 10];
	Float3 color = center.xyz;
	ushort mask = center.w;

	// Load neighbour color, get max min and guassian
	Float3 neighbourMax = Float3(FLT_MIN);
	Float3 neighbourMin = Float3(FLT_MAX);
	Float3 gaussianColor = 0;
	#pragma unroll
	for (int i = 0; i < 3; ++i)
	{
		#pragma unroll
		for (int j = 0; j < 3; ++j)
		{
			Float3 neighbourColor = sharedBuffer[threadIdx.x + j + (threadIdx.y + i) * 10].xyz;

			// gaussian color
			gaussianColor += Gaussian3x3[i * 3 + j] * neighbourColor;

			// max min
			neighbourColor = RgbToYcocg(neighbourColor);
			neighbourMax = max3f(neighbourMax, neighbourColor);
			neighbourMin = min3f(neighbourMin, neighbourColor);
		}
	}

    // sample history color
	Float2 motionVec = Load2DHalf2(motionVectorBuffer, Int2(x, y)) - Float2(0.5f);
	Float2 uv = (Float2(x, y) + 0.5f) * (1.0f / Float2(size.x, size.y));
	Float2 historyUv = uv + motionVec;

	// history uv out of screen
	if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0 || historyUv.y > 1.0)
	{
		Store2DHalf3Ushort1( { gaussianColor, mask } , colorBuffer, Int2(x, y));
		return;
	}

	// sample history
	Float3 colorHistory = SampleBicubicSmoothStep(accumulateBuffer, Load2DHalf3Ushort1Float3, historyUv, historySize);

	// clamp history
	Float3 colorHistoryYcocg = RgbToYcocg(colorHistory);
	colorHistoryYcocg = clamp3f(colorHistoryYcocg, neighbourMin, neighbourMax);
	colorHistory = YcocgToRgb(colorHistoryYcocg);

	// float lumaHistory = colorHistoryYcocg.x;
	// float lumaMin = neighbourMin.x;
	// float lumaMax = neighbourMax.x;
	// float lumaCurrent = GetLuma(color);

	// load history normal and depth
	Int2 nearestHistoryIdx = Int2(historyUv.x * historySize.x, historyUv.y * historySize.y);
	Float2 normalAndDepthHistory = Load2DFloat2(normalDepthHistoryBuffer, nearestHistoryIdx);
	Float3 normalHistory = DecodeNormal_R11_G10_B11(normalAndDepthHistory.x);
	float depthHistory = normalAndDepthHistory.y;
	ushort maskHistory = Load2DHalf3Ushort1(accumulateBuffer, nearestHistoryIdx).w;

	// discard history
	float depthRatio = SafeDivide(depthHistory, depth);
	const float depthRatioUpperLimit = 1.2f;
	const float depthRatioLowerLimit = 1.0f / depthRatioUpperLimit;

	bool discardHistory =
		(depthHistory >= RayMax) ||
		(depthRatio > depthRatioUpperLimit) ||
		(depthRatio < depthRatioLowerLimit) ||
		(mask != maskHistory);

	Float3 outColor;
	if (discardHistory)
	{
		// use gaussian color if history is rejected
		outColor = gaussianColor;
	}
	else
	{
		// blend factor base
		float blendFactor = 1.0f / 16.0f;

		// history noies level
		//blendFactor = lerpf(1.0f / 16.0f, 1.0f / 64.f, noiseLevel);

		// anti flickering
		//blendFactor *= 0.2f + 0.8f * clampf(0.5f * min( abs(lumaHistory - lumaMin), abs(lumaHistory - lumaMax) ) / max3( lumaHistory, lumaCurrent, 1e-4f ));

		// weight with luma hdr factor
		// float weightA = blendFactor * max(0.0001f, 1.0f / (GetLuma(color) + 4.0f));
		// float weightB = (1.0f - blendFactor) * max(0.0001f, 1.0f / (GetLuma(colorHistory) + 4.0f));
		// float weightSum = SafeDivide(1.0f, weightA + weightB);
		// weightA *= weightSum;
		// weightB *= weightSum;

		// blend
		//outColor = color * weightA + colorHistory * weightB
		outColor = color * blendFactor + colorHistory * (1.0f - blendFactor);
	}

	if (isnan(outColor.x) || isnan(outColor.y) || isnan(outColor.z))
    {
        printf("TemporalFilter: nan found at (%d, %d)\n", x, y);
        outColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { outColor, mask } , colorBuffer, Int2(x, y));
}