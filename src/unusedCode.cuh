
#pragma once

#include <cuda_runtime.h>
#include "sampler.cuh"

__global__ void BufferCopy(
	SurfObj   inBuffer,
	SurfObj   OutBuffer,
	Int2                  size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

    Store2D(Load2D(inBuffer, idx), OutBuffer, idx);
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

__global__ void Histogram(
	float*    histogram,
	SurfObj   InBuffer,
	Int2      size)
{
	__shared__ float sharedHistogram[8][8][64];
	Int2 threadId(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	for (int i = 0; i < 64; ++i)
	{
		sharedHistogram[threadIdx.x][threadIdx.y][i] = 0;
	}

	// shared histogram of 8x8 block
	Float3 color = Load2D(InBuffer, threadId).xyz;

	//float luminance = color.max();
	float luminance = color.x * 0.3 + color.y * 0.6 + color.z * 0.1;
	float logLuminance = log2f(luminance) * 0.1 + 0.9;

	float fBucket = clampf(logLuminance, 0.0, 1.0) * 63 * 0.99999;

	uint bucket0 = (uint)fBucket;
	uint bucket1 = bucket0 + 1;

	float bucketWeight0 = fmodf(fBucket, 1.0);
	float bucketWeight1 = 1.0 - bucketWeight0;

	sharedHistogram[threadId.x][threadId.y][bucket0] += bucketWeight0;
	sharedHistogram[threadId.x][threadId.y][bucket1] += bucketWeight1;

	// gather y dimension
	#define unroll
	for (int i = 0; i < 3; ++i)
	{
		int offset = 1 << i;
		if (threadId.y % (offset * 2) == 0)
		{
			#define unroll
			for (int j = 0; j < 16; ++j)
			{
				sharedHistogram[threadId.x][threadId.y][j * 4 + 0] += sharedHistogram[threadId.x][threadId.y + offset][j * 4 + 0];
				sharedHistogram[threadId.x][threadId.y][j * 4 + 1] += sharedHistogram[threadId.x][threadId.y + offset][j * 4 + 1];
				sharedHistogram[threadId.x][threadId.y][j * 4 + 2] += sharedHistogram[threadId.x][threadId.y + offset][j * 4 + 2];
				sharedHistogram[threadId.x][threadId.y][j * 4 + 3] += sharedHistogram[threadId.x][threadId.y + offset][j * 4 + 3];
			}
		}
	}

	// gather x dimension
	#define unroll
	for (int i = 0; i < 3; ++i)
	{
		int offset = 1 << i;
		if (threadId.x % (offset * 2) == 0)
		{
			#define unroll
			for (int j = 0; j < 16; ++j)
			{
				sharedHistogram[threadId.x][threadId.y][j * 4 + 0] += sharedHistogram[threadId.x + offset][threadId.y][j * 4 + 0];
				sharedHistogram[threadId.x][threadId.y][j * 4 + 1] += sharedHistogram[threadId.x + offset][threadId.y][j * 4 + 1];
				sharedHistogram[threadId.x][threadId.y][j * 4 + 2] += sharedHistogram[threadId.x + offset][threadId.y][j * 4 + 2];
				sharedHistogram[threadId.x][threadId.y][j * 4 + 3] += sharedHistogram[threadId.x + offset][threadId.y][j * 4 + 3];
			}
		}
	}

	// iterate through all 8x8 blocks
	if (threadId.x == 0 && threadId.y == 0)
	{
		for (int i = 0; i < gridDim.x; ++i)
		{
			for (int j = 0; j < gridDim.y; ++i)
			{
				#define unroll
				for (int k = 0; k < 16; ++k)
				{
					histogram[k * 4 + 0] += sharedHistogram[0][0][k * 4 + 0];
					histogram[k * 4 + 1] += sharedHistogram[0][0][k * 4 + 1];
					histogram[k * 4 + 2] += sharedHistogram[0][0][k * 4 + 2];
					histogram[k * 4 + 3] += sharedHistogram[0][0][k * 4 + 3];
				}
			}
		}
	}
}

__global__ void TemporalFilter(
	SurfObj   colorBuffer, // [in/out]
	SurfObj   accumulateBuffer,
	Int2      size)
{
	// index for pixel 14 x 14
	Int2 idx2;
	idx2.x = blockIdx.x * 14 + threadIdx.x - 1;
	idx2.y = blockIdx.y * 14 + threadIdx.y - 1;

	// index for shared memory buffer
	Int2 idx3;
	idx3.x = threadIdx.x;
	idx3.y = threadIdx.y;

	Float4 currentColor = Load2D(colorBuffer, idx2);
	Float4 historyColor = Load2D(accumulateBuffer, idx2);
	Float3 normalValue = DecodeNormal_R11_G10_B11(currentColor.w);

	__shared__ Float4 sharedBuffer[16][16];
	sharedBuffer[threadIdx.x][threadIdx.y] = currentColor;
	__syncthreads();

	// Border margin pixel finish work
	if (idx3.x < 1 || idx3.y < 1 || idx3.x > 14 || idx3.y > 14) { return; }
	// Early return for background pixel
	if (fabsf(normalValue.x) < 0.01 && fabsf(normalValue.y) < 0.01 && fabsf(normalValue.z) < 0.01) { return; }

	// current max min
	Float3 colorMax = Float3(FLT_MIN);
	Float3 colorMin = Float3(FLT_MAX);
	for (int i = 0; i < 9; ++i)
	{
		Int2 uv(threadIdx.x + (i % 3) - 1, threadIdx.y + (i / 3) - 1);
		Float4 tmp = sharedBuffer[uv.x][uv.y];
		Float3 color = tmp.xyz;
		colorMax = max3f(colorMax, color);
		colorMin = min3f(colorMax, color);
	}

	// clamp history
	Float3 history = historyColor.xyz;
	history = clamp3f(history, colorMin, colorMax);

	// blend
	float blendFactor = 1.0f / 16.0f;
	Float3 finalColor = lerp3f(history, currentColor.xyz, blendFactor);

	Store2D(Float4(finalColor, currentColor.w), colorBuffer, idx2);
}

__global__ void TemporalDenoisingFilter(
	SurfObj   colorBuffer, // [in/out]
	SurfObj   accumulateBuffer,
	Int2      size)
{
	__shared__ Float4 sharedBuffer[32][32];
	__shared__ Float4 sharedBufferHistory[32][32];

	// index for pixel 28 x 28
	Int2 idx2;
	idx2.x = blockIdx.x * 28 + threadIdx.x - 2;
	idx2.y = blockIdx.y * 28 + threadIdx.y - 2;

	// index for shared memory buffer
	Int2 idx3;
	idx3.x = threadIdx.x;
	idx3.y = threadIdx.y;

	// read global memory buffer
	Float4 colorNormal = Load2D(colorBuffer, idx2);
	Float3 colorValue = colorNormal.xyz;
	float currentNormal = colorNormal.w;
	Float3 normalValue = DecodeNormal_R11_G10_B11(colorNormal.w);

	Float4 historyColorNormal = Load2D(accumulateBuffer, idx2);
	Float3 historyColorValue = historyColorNormal.xyz;
	//float historyNormal = historyColorNormal.w;
	Float3 historyNormalValue = DecodeNormal_R11_G10_B11(historyColorNormal.w);

	// -------------------------------- find corresponding history location --------------------------------

	// -------------------------------- load lds with history --------------------------------
	sharedBufferHistory[threadIdx.x][threadIdx.y] = historyColorNormal;
	sharedBuffer[threadIdx.x][threadIdx.y] = colorNormal;
	__syncthreads();

	// Border margin pixel finish work
	if (idx3.x < 2 || idx3.y < 2 || idx3.x > 29 || idx3.y > 29) { return; }
	// Early return for background pixel
	if (fabsf(normalValue.x) < 0.01 && fabsf(normalValue.y) < 0.01 && fabsf(normalValue.z) < 0.01) { return; }

	// -------------------------------- deviation based on history--------------------------------
	float colorHistoryDeviation = 0;
	float normalHistoryDeviation = 0;
	for (int i = 0; i < 25; ++i)
	{
		Int2 uv(threadIdx.x + (i % 5) - 2, threadIdx.y + (i / 5) - 2);

		Float4 bufferReadTmp = sharedBufferHistory[uv.x][uv.y];
		Float3 historyColor = bufferReadTmp.xyz;
		Float3 historyNormal = DecodeNormal_R11_G10_B11(bufferReadTmp.w);

		Float3 t = colorValue - historyColor;
		float dist2 = dot(t,t);
		colorHistoryDeviation += dist2 * filterKernel[i];

		t = normalValue - historyNormal;
        dist2 = dot(t,t);
        normalHistoryDeviation += dist2 * filterKernel[i];
	}

	// -------------------------------- current neighbour median --------------------------------
	Float3 neighbor3x3[9];
	float colorDeviation = 0;
	for (int i = 0; i < 9; ++i)
	{
		Int2 uv(threadIdx.x + (i % 3) - 1, threadIdx.y + (i / 3) - 1);

		Float4 bufferReadTmp = sharedBuffer[uv.x][uv.y];
		Float3 color = bufferReadTmp.xyz;
		neighbor3x3[i] = color;

		Float3 t = colorValue - color;
		float dist2 = dot(t,t);
		colorDeviation += dist2 * filterKernel[i];
	}
	sort9(neighbor3x3);
	Float3 medianAverage = (neighbor3x3[4] + neighbor3x3[5]) / 2.0f;

	// -------------------------------- blend for current color --------------------------------
	const float maxDeviationTolerance = 0.3;
	float blendFactor = clampf(2.0f - expf(pow3(1.0f - (colorHistoryDeviation) / maxDeviationTolerance))); // https://graphtoy.com/?f1(x)=2-exp(pow(1-x/0.5,3))
	Float3 currentColor = lerp3f(colorValue, medianAverage, blendFactor);
	currentColor = colorValue;

	// -------------------------------- blending with history --------------------------------

	// https://graphtoy.com/?f1(x)=exp(2*pow(x,2))+0.1-1
	float blendFactorNormal = expf(2.0f * pow2(normalHistoryDeviation)) - 1.0f + 1.0f/16.0f;

	// https://graphtoy.com/?f1(x)=exp(-16*pow(x,2))*0.45+1/16
	float blendFactorColor = expf(-16.0f * pow2(colorHistoryDeviation)) * 0.45 + 1.0f/16.0f;

	// blending factor
	blendFactor = clampf(lerpf(blendFactorNormal, blendFactorColor, 0.5f));
	//blendFactor = 1.0f/16.0f;

	// history blend
	Float3 finalColor = lerp3f(historyColorValue, currentColor, blendFactor);

	// write out
	Store2D(Float4(finalColor, currentNormal), colorBuffer, idx2);


#if 0
	// gather 5x5 average
	Float3 average = Float3(0.0);
	for (int i = 0; i < 25; ++i)
	{
		Int2 uv(threadIdx.x + (i % 5) - 2, threadIdx.y + (i / 5) - 2);
		Float4 bufferReadTmp = sharedBuffer[uv.x][uv.y];
		Float3 color = bufferReadTmp.xyz;
		Float3 normal = DecodeNormal_R11_G10_B11(bufferReadTmp.w);
		Float3 t = normalValue - normal;
        float dist2 = max1f(dot(t,t), 0.0);
        float normalWeight = min1f(expf(-(dist2) / 1.0), 1.0);

		float weight = normalWeight;
		sumOfWeight += weight;
		average += weight * color;
	}
	average /= sumOfWeight;
	// gather 5x5 standard deviation
	Float3 stddev = Float3(0.0);
	for (int i = 0; i < 25; ++i)
	{
		Int2 uv(threadIdx.x + (i % 5) - 2, threadIdx.y + (i / 5) - 2);
		Float4 bufferReadTmp = sharedBuffer[uv.x][uv.y];
		Float3 color = bufferReadTmp.xyz;
		Float3 normal = DecodeNormal_R11_G10_B11(bufferReadTmp.w);
		Float3 t = normalValue - normal;
        float dist2 = max1f(dot(t,t), 0.0);
        float normalWeight = min1f(expf(-(dist2) / 1.0), 1.0);

		Float3 temp = (color - average) * normalWeight;
	 	stddev += temp * temp;
	}
	stddev = sqrt3f(stddev / (sumOfWeight - 0.99));
	// set outliers to average
	Float3 diff = abs(colorValue - average);
	Float3 isNoise = diff - stddev;
	Float3 averageAround = (average * sumOfWeight - colorValue) / (sumOfWeight - 0.99);

	colorValue = Float3(
		isNoise.x > 0 ? averageAround.x : colorValue.x,
		isNoise.y > 0 ? averageAround.y : colorValue.y,
		isNoise.z > 0 ? averageAround.z : colorValue.z);

	sharedBuffer[threadIdx.x][threadIdx.y] = Float4(colorValue, colorNormal.w);

	Float3 sumOfColor = Float3(0.0);
	sumOfWeight = 0.0;
#endif

}

__global__ void TAA(
	SurfObj   currentBuffer,
	SurfObj   accumulateBuffer,
	Int2      size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float4 currentColor = Load2D(currentBuffer, idx);
	Float4 historyColor = Load2D(accumulateBuffer, idx);

	float blendFactor = 1.0f / 16.0f;
	Float3 outColor = currentColor.xyz * blendFactor + historyColor.xyz * (1.0f - blendFactor);

    Store2D(Float4(outColor, currentColor.w), currentBuffer, idx);
	Store2D(Float4(outColor, currentColor.w), accumulateBuffer, idx);
}
