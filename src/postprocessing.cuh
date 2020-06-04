
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
	float* exposure)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	const float W = 11.2;

	Float3 texColor = Load2D(colorBuffer, idx).xyz;
	texColor *= exposure[0];

	float ExposureBias = 2.0f;
	Float3 curr = Uncharted2Tonemap(ExposureBias * texColor);

	Float3 whiteScale = 1.0f/Uncharted2Tonemap(W);
	Float3 color = curr * whiteScale;

	Float3 retColor = clamp3f(pow3f(color, 1.0f / 2.2f), Float3(0), Float3(1));

	Store2D(Float4(retColor, 1.0), colorBuffer, idx);
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

	float blendFactor = 0.1f;
	Float3 outColor = currentColor.xyz * blendFactor + historyColor.xyz * (1.0f - blendFactor);

    Store2D(Float4(outColor, currentColor.w), currentBuffer, idx);
	Store2D(Float4(outColor, currentColor.w), accumulateBuffer, idx);
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

//__global__ void BufferCopyFp16(
//	SurfObj   OutBuffer,
//	SurfObj   inBuffer,
//	Int2                  size)
//{
//	Int2 idx;
//	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
//	idx.y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (idx.x >= size.x || idx.y >= size.y) return;
//
//    Store2D(Load2Dfp16(inBuffer, idx), OutBuffer, idx);
//}

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

#if 0
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
#endif

__global__ void Histogram2(
	uint*     histogram,
	SurfObj   InBuffer,
	Int2      size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 color = Load2D(InBuffer, idx).xyz;
	float luminance = color.max();//dot(color, Float3(0.3, 0.6, 0.1));
	float logLuminance = log2f(luminance) * 0.1 + 0.75;
	uint fBucket = (uint)rintf(clampf(logLuminance, 0.0, 1.0) * 63 * 0.99999);
	atomicInc(&histogram[fBucket], size.x * size.y);
}

__device__ __inline__ float BinToLum(int i) { return exp2f(((float)i / (63 * 0.99999) - 0.75) / 0.1); }

__global__ void AutoExposure(float* exposure, uint* histogram, float area, float deltaTime)
{
	// Threshold for histogram cut
	const float darkThreshold = 0.4;
	const float brightThreshold = 0.9;

	// sum of lum and area
	float lumiSum = 0;
	float lumiSumArea = 0;

	// accumulate histogram area
	float accuHistArea = 0;

	// bright 20% lum
	float brightLum = 0;

	// first loop for dark cut line
	int i = 0;
	for (; i < 64; ++i)
	{
		// read histogram value
		uint hist = histogram[i];

		// convert to float percentage
		float fHist = (float)hist / area;

		// find lum by bin index
		float lum = BinToLum(i);

		// add to accu hist area, test for dark cut line
		accuHistArea += fHist;
		float dark = accuHistArea - darkThreshold;

		// add the part fall in the area we want
		if (dark > 0)
		{
			lumiSumArea += dark;
			lumiSum += dark * lum;
			break;
		}
	}

	// second part of loop for bright cut line
	for (; i < 64; ++i)
	{
		// same as above
		uint hist = histogram[i];
		float fHist = (float)hist / area;
		float lum = BinToLum(i);

		accuHistArea += fHist;
		float bright = accuHistArea - brightThreshold;

		if (bright > 0)
		{
			float partial = brightThreshold - (accuHistArea - fHist);
			lumiSumArea += partial;
			lumiSum += partial * lum;
			brightLum = lum;
			break;
		}
		else
		{
			lumiSumArea += fHist;
			lumiSum += fHist * lum;
		}
	}

	// get average lum
	float aveLum = lumiSum / lumiSumArea;

	// clamp to min and max. Note 0.1 is no good but for lower night EV value
	aveLum = clampf(aveLum, 0.1, 10.0);

	// read history lum
	float lumTemp = exposure[1];
	float lumBrightTemp = exposure[2];

	// perform eye adaption (smooth transit between EV)
	lumTemp = lumTemp + (aveLum - lumTemp) * (1.0 - expf(-deltaTime * 0.001));
	lumBrightTemp = lumBrightTemp + (brightLum - lumBrightTemp) * (1.0 - expf(-deltaTime * 0.001));

	// Exposure compensation curve: Brigher for day, darker for night
	float EC = 1.03 - 2.0 / (log10f(lumTemp + 1.0) + 2.0);

	// Standart exposure value compute with a constant gain
	float EV = 7.0 * EC / lumTemp;

	// write EV and lum
	exposure[0] = EV;
	exposure[1] = lumTemp;
	exposure[2] = lumBrightTemp;
}

__global__ void DownScale4(SurfObj InBuffer, SurfObj OutBuffer, Int2 size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ Float4 sharedBuffer[8][8];
	sharedBuffer[threadIdx.x][threadIdx.y] = Load2D(InBuffer, idx) / 16;

	if (threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0)
	{
		sharedBuffer[threadIdx.x][threadIdx.y] =
			sharedBuffer[threadIdx.x][threadIdx.y] +
			sharedBuffer[threadIdx.x + 1][threadIdx.y] +
			sharedBuffer[threadIdx.x][threadIdx.y + 1] +
			sharedBuffer[threadIdx.x + 1][threadIdx.y + 1];
	}

	if (threadIdx.x % 4 == 0 && threadIdx.y % 4 == 0)
	{
		Float4 result =
			sharedBuffer[threadIdx.x][threadIdx.y] +
			sharedBuffer[threadIdx.x + 2][threadIdx.y] +
			sharedBuffer[threadIdx.x][threadIdx.y + 2] +
			sharedBuffer[threadIdx.x + 2][threadIdx.y + 2];
		Store2D(result, OutBuffer, Int2(idx.x / 4, idx.y / 4));
	}
}

__constant__ float filterKernel[25] = {
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0,  3.0 / 32.0, 3.0 / 128.0,
		1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
		1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };

__global__ void DenoiseKernel(
	SurfObj   colorBuffer, // [in/out]
	Int2      size)
{
	// index for pixel 28 x 28
	Int2 idx2;
	idx2.x = blockIdx.x * 28 + threadIdx.x - 2;
	idx2.y = blockIdx.y * 28 + threadIdx.y - 2;

	// index for shared memory buffer
	Int2 idx3;
	idx3.x = threadIdx.x;
	idx3.y = threadIdx.y;

	// read global memory buffer. One-to-one mapping
	Float4 bufferRead = Load2D(colorBuffer, idx2);
	Float3 colorValue = bufferRead.xyz;
	Float3 normalValue = DecodeNormal_R11_G10_B11(bufferRead.w);

	// Save global memory to shared memory. One-to-one mapping
	__shared__ Float4 sharedBuffer[32][32];
	sharedBuffer[threadIdx.x][threadIdx.y] = bufferRead;
	__syncthreads();

	// Border margin pixel finish work
	if (idx3.x < 2 || idx3.y < 2 || idx3.x > 29 || idx3.y > 29) { return; }

	// Early return for background pixel
	if (fabsf(normalValue.x) < 0.01 && fabsf(normalValue.y) < 0.01 && fabsf(normalValue.z) < 0.01) { return; }

	float sumOfWeight = 0.0;

#if 1

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

	sharedBuffer[threadIdx.x][threadIdx.y] = Float4(colorValue, bufferRead.w);

#endif

	Float3 sumOfColor = Float3(0.0);
	sumOfWeight = 0.0;

	// atrous
	__syncthreads();
	for (int i = 0; i < 25; ++i)
	{
		Int2 uv(threadIdx.x + (i % 5) - 2, threadIdx.y + (i / 5) - 2);
		Float4 bufferReadTmp = sharedBuffer[uv.x][uv.y];
		Float3 color = bufferReadTmp.xyz;
		Float3 normal = DecodeNormal_R11_G10_B11(bufferReadTmp.w);

		Float3 t = normalValue - normal;
        float dist2 = max1f(dot(t,t), 0.0);
        float normalWeight = min1f(expf(-(dist2) / 0.1), 1.0);

        t = colorValue - color;
        dist2 = dot(t,t);
        float colorWeight = min1f(expf(-(dist2) / 10.0), 1.0);

		float weight = colorWeight * normalWeight * filterKernel[i];
        sumOfColor += color * weight;
        sumOfWeight += weight;
	}

	// final output
	Store2D(Float4(sumOfColor / sumOfWeight, 1.0), colorBuffer, idx2);
}

__global__ void BloomGuassian(SurfObj outBuffer, SurfObj inBuffer, Int2 size, float* exposure)
{
	// index for pixel 28 x 28
	Int2 idx2;
	idx2.x = blockIdx.x * 12 + threadIdx.x - 2;
	idx2.y = blockIdx.y * 12 + threadIdx.y - 2;

	// read global memory buffer. One-to-one mapping
	Float3 inColor = Load2D(inBuffer, idx2).xyz;

	// discard dark pixels
	float lum = inColor.max();
	float brightLum = exposure[2];
	//inColor = lum > brightLum ? inColor : 0;
	inColor *= max(sqrtf(lum * lum - brightLum), 0.0);

	// Save global memory to shared memory. One-to-one mapping
	__shared__ Float3 sharedBuffer[16][16];
	sharedBuffer[threadIdx.x][threadIdx.y] = inColor;

	// Border margin pixel finish work
	if (threadIdx.x < 2 || threadIdx.y < 2 || threadIdx.x > 13 || threadIdx.y > 13) { return; }

	Float3 outColor = 0;
	__syncthreads();
	for (int i = 0; i < 25; ++i)
	{
		int x = threadIdx.x + (i % 5) - 2;
		int y = threadIdx.y + (i / 5) - 2;
		outColor += filterKernel[i] * sharedBuffer[x][y];
	}

	Store2D(Float4(outColor, 1.0), outBuffer, idx2);
}

__global__ void Bloom(SurfObj colorBuffer, SurfObj buffer4, SurfObj buffer16, Int2 size, Int2 size4, Int2 size16)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx.x >= size.x || idx.y >= size.y) return;
	Float2 uv((float)idx.x / size.x, (float)idx.y / size.y);

	Float3 sampledColor4 = SampleBicubicCatmullRom(buffer4, uv, size4).xyz;
	Float3 sampledColor16 = SampleBicubicCatmullRom(buffer16, uv, size16).xyz;
	Float3 color = Load2D(colorBuffer, idx).xyz;

	color = color + (sampledColor4 + sampledColor16) * 0.05;

	Store2D(Float4(color, 1.0), colorBuffer, idx);
}

inline __device__ float LensFlareRand(float w)
{
    return fract(sinf(w) * 1000.0);
}

inline __device__ float LensFlareRegShape(Float2 p, int N)
{
	float a = atan2f(p.x, p.y) + 0.2;
	float b = TWO_PI / float(N);
	float f = smoothstep1f(0.5, 0.51, cosf(floorf(0.5 + a / b) * b - a) * p.length());
    return f;
}

__device__ Float3 LensFlareCircle(Float2 p, float size, float dist, Float2 mouse)
{
	float l = length(p + mouse * (dist * 4.0)) + size / 2.0;

	float c  = max(0.01 - powf(length(p + mouse * dist), size * 1.4), 0.0) * 30.0; // big circle
	float c1 = max(0.001 - powf(l - 0.3, 1.0 / 40.0) + sinf(l * 30.0), 0.0) * 3.0; // ring
	float c2 = max(0.04 / powf(length(p - mouse * dist / 2.0 + 0.09) * 1.0, 1.0), 0.0) / 20.0; // dot
	float s  = max(0.01 - powf(LensFlareRegShape(p * 5.0 + mouse * dist * 5.0 + 0.9, 6), 1.0), 0.0) * 6.0; // Hexagon

	Float3 color = cos3f(Float3(0.44, 0.24, 0.2) * 8.0 + dist * 4.0) * 0.5 + 0.5;

	Float3 f = c * color;
	f += c1 * color;
	f += c2 * color;
	f += s * color;

	return f - 0.01;
}

__global__ void LensFlare(Float2 sunPos, SurfObj colorBuffer, Int2 size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx.x >= size.x || idx.y >= size.y) return;

	Float2 uv((float)idx.x / (float)size.x, (float)idx.y / (float)size.y);
	uv -= Float2(0.5);
	uv.x *= (float)size.x / (float)size.y;

	Float3 color = Load2D(colorBuffer, idx).xyz;

	Float2 vec = uv - sunPos;
	float len = vec.length();

	for (float i = 0.0f; i < 10.0f; i++)
	{
        color += LensFlareCircle(uv,
			                     powf(LensFlareRand(i * 2000.0f) * 1.8f, 2.0f) + 1.41f,
			                     LensFlareRand(i * 20.0f) * 3.0f + 0.2f - 0.5f,
			                     sunPos);
    }

	// get angle and length of the sun (uv - mouse)
    float angle = atan2f(vec.y, vec.x);

    // add the sun with the frill things
    //color += max(0.1 / powf(len * 50. , 5.0)       , 0.0) * abs(sinf(angle * 5.0 + cosf(angle * 9.0))) / 20.0;
    color += max(0.1 / powf(len * 10.0, 1.0 / 20.0), 0.0) + abs(sinf(angle * 3.0 + cosf(angle * 9.0))) / 16.0 * abs(sinf(angle * 9.0));

	Store2D(Float4(color, 1.0), colorBuffer, idx);
}