#pragma once

#include "kernel.cuh"
#include "sampler.cuh"

//----------------------------------------------------------------------------------------------
//------------------------------------- Temporal Filter ---------------------------------------
//----------------------------------------------------------------------------------------------

__constant__ float Gaussian3x3[9] = {
    1.0f/16.0f, 1.0f/8.0f, 1.0f / 16.0f,
	1.0f / 8.0f, 1.0f/4.0f, 1.0f / 8.0f,
	1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f,
};

__global__ void TemporalFilter(
	SurfObj   colorBuffer,
	SurfObj   accumulateBuffer,
	SurfObj   normalDepthBuffer,
	SurfObj   normalDepthHistoryBuffer,
    SurfObj   motionVectorBuffer,
	Int2      size,
	Int2      historySize)
{
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

	// Early return for background pixel
	if (depth >= RayMax)
	{
		Store2DHalf3Ushort1( { color, mask } , colorBuffer, Int2(x, y));
		return;
	}

	// Load neighbour color, get max min and guassian
	Float3 neighbourMax = color;
	Float3 neighbourMin = color;
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

			// neighbour only
			if (i != 1 && j != 1)
			{
				// max min
				neighbourMax = max3f(neighbourMax, neighbourColor);
				neighbourMin = min3f(neighbourMin, neighbourColor);
			}
		}
	}

    // sample history color
	Float2 motionVec = Load2DHalf2(motionVectorBuffer, Int2(x, y)) - Float2(0.5f);
	Float2 uv = (Float2(x, y) + 0.5f) * (1.0f / Float2(size.x, size.y));
	Float2 historyUv = uv + motionVec;
    Float3 colorHistory;

	// history uv out of screen
	if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0 || historyUv.y > 1.0)
	{
		Store2DHalf3Ushort1( { gaussianColor, mask } , colorBuffer, Int2(x, y));
		return;
	}

	// sample history
	colorHistory = SampleBicubicSmoothStep(accumulateBuffer, Load2DHalf3Ushort1Float3, historyUv, historySize);

	// clamp history
	colorHistory = clamp3f(colorHistory, neighbourMin, neighbourMax);

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

	// blending factor
	const float blendFactor = 1.0f / 16.0f;

	Float3 outColor;
	if (discardHistory)
	{
		outColor = gaussianColor;
	}
	else
	{
		// blending
		outColor = color * blendFactor + colorHistory * (1.0f - blendFactor);
	}

	if (isnan(outColor.x) || isnan(outColor.y) || isnan(outColor.z))
    {
        printf("nan found at (%d, %d)\n", x, y);
        outColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { outColor, mask } , colorBuffer, Int2(x, y));
}

//----------------------------------------------------------------------------------------------
//------------------------------------- Atous Filter ---------------------------------------
//----------------------------------------------------------------------------------------------

extern __constant__ float filterKernel[25];

__global__ void AtousFilter2(
	SurfObj   colorBuffer,
	SurfObj   accumulateBuffer,
	SurfObj   normalDepthBuffer,
	SurfObj   normalDepthHistoryBuffer,
	SurfObj   sampleCountBuffer,
	Int2      size)
{
	struct AtrousLDS
	{
		Float3 color;
		float depth;
		Float3 normal;
		ushort mask;
	};
	__shared__ AtrousLDS sharedBuffer[20 * 20];

	// calculate address
	int x = threadIdx.x + blockIdx.x * 16;
	int y = threadIdx.y + blockIdx.y * 16;

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

	if (x >= size.x || y >= size.y) return;

	// load center
	AtrousLDS center   = sharedBuffer[threadIdx.x + 2 + (threadIdx.y + 2) * 20];
	Float3 colorValue  = center.color;
	float depthValue   = center.depth;
	Float3 normalValue = center.normal;
	ushort maskValue   = center.mask;

	if (depthValue >= RayMax) { return; }

	// -------------------------------- atrous filter --------------------------------
	// Reference: https://jo.dreggn.org/home/2010_atrous.pdf
	Float3 sumOfColor = 0;
	float sumOfWeight = 0;
	Float2 variancePair = 0;
	#pragma unroll
	for (int i = 0; i < 5; ++i)
	{
		#pragma unroll
		for (int j = 0; j < 5; ++j)
		{
			AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + j + (threadIdx.y + i) * 20];

			// get data
			Float3 color  = bufferReadTmp.color;
			float depth   = bufferReadTmp.depth;
			Float3 normal = bufferReadTmp.normal;
			ushort mask   = bufferReadTmp.mask;

			// variance pair
			float lum = color.getmax();
			variancePair += Float2(lum, lum * lum);

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
	Float3 finalColor = sumOfColor / sumOfWeight;

	// variance
	variancePair /= 25.0f;
	float variance = max(0.0f, variancePair.y - variancePair.x * variancePair.x);

	//
	if (((variance > 0.0001f) || (variancePair.x < 0.1f)) && ((x & 0x7) == 3) && ((y & 0x7) == 3))
	{
		Int2 gridLocation = Int2(x >> 3, y >> 3);
		Store2D_uchar1(2, sampleCountBuffer, gridLocation);
	}

	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        printf("nan found at (%d, %d)\n", x, y);
        finalColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { finalColor, maskValue } , colorBuffer, Int2(x, y));

	// store to history
	Store2DHalf3Ushort1( { finalColor, maskValue } , accumulateBuffer, Int2(x, y));
	Store2DFloat2( { EncodeNormal_R11_G10_B11(normalValue), depthValue } , normalDepthHistoryBuffer, Int2(x, y));
}
