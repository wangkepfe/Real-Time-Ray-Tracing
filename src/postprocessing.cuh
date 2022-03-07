
#pragma once

#include <cuda_runtime.h>
#include "sampler.cuh"
#include "geometry.cuh"
#include "precision.cuh"
#include "settingParams.h"
#include "gaussian.cuh"
#include "color.h"

#define USE_CATMULL_ROM_SAMPLER 0
#define USE_BICUBIC_SMOOTH_STEP_SAMPLER 1

__device__ __forceinline__ float GetLuma(const Float3& rgb)
{
	return rgb.y * 2.0f + rgb.x + rgb.z;
}

//----------------------------------------------------------------------------------------------
//------------------------------------- Histogram and Auto Exposure ---------------------------------------
//----------------------------------------------------------------------------------------------

__global__ void Histogram2(
	uint     *histogram,
	SurfObj   InBuffer,
	Int2      size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 color = Load2DHalf4(InBuffer, idx).xyz;
	float luminance = dot(color, Float3(0.3, 0.6, 0.1));
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
	aveLum = clampf(aveLum, 0.1, 100.0);

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

//----------------------------------------------------------------------------------------------
//------------------------------------- Downscale ---------------------------------------
//----------------------------------------------------------------------------------------------

__global__ void DownScale4(SurfObj InBuffer, SurfObj OutBuffer, Int2 size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ Float4 sharedBuffer[8][8];
	sharedBuffer[threadIdx.x][threadIdx.y] = Load2DHalf4(InBuffer, idx) / 16;

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

		Store2DHalf4(result, OutBuffer, Int2(idx.x / 4, idx.y / 4));
	}
}

//----------------------------------------------------------------------------------------------
//------------------------------------- Median Filter ---------------------------------------
//----------------------------------------------------------------------------------------------

struct LumaIdxPair
{
	float luma;
	int idx;
};

__device__ __inline__ LumaIdxPair min(const LumaIdxPair& a, const LumaIdxPair& b) { return (a.luma < b.luma) ? a : b; }
__device__ __inline__ LumaIdxPair max(const LumaIdxPair& a, const LumaIdxPair& b) { return (a.luma > b.luma) ? a : b; }

__device__ __inline__ LumaIdxPair min3(const LumaIdxPair& v1, const LumaIdxPair& v2, const LumaIdxPair& v3) { return min(min(v1, v2), v3); }
__device__ __inline__ LumaIdxPair max3(const LumaIdxPair& v1, const LumaIdxPair& v2, const LumaIdxPair& v3) { return max(max(v1, v2), v3); }

__device__ __inline__ void swap(LumaIdxPair& a, LumaIdxPair& b) { LumaIdxPair temp = a; a = b; b = temp; }
__device__ __inline__ void sort2(LumaIdxPair& a, LumaIdxPair& b) { if (a.luma > b.luma) { swap(a, b); } }
__device__ __inline__ void sort3(LumaIdxPair& v1, LumaIdxPair& v2, LumaIdxPair& v3) { sort2(v2, v3); sort2(v1, v2); sort2(v2, v3); }

__device__ __inline__ void sort9(LumaIdxPair* p)
{
    sort3(p[0], p[1], p[2]);
    sort3(p[3], p[4], p[5]);
    sort3(p[6], p[7], p[8]);

    p[6] = max3(p[0], p[3], p[6]);
    p[2] = min3(p[2], p[5], p[8]);

    sort3(p[1], p[4], p[7]);
    sort3(p[2], p[4], p[6]);
}

__global__ void MedianFilter(
	SurfObj   colorBuffer, // [in/out]
	SurfObj   normalDepthBuffer,
	SurfObj   accumulateBuffer,
	Int2      size)
{
	struct LDS
	{
		Float3 color;
		float depth;
		Float3 normal;
		ushort mask;
	};

	const int blockdim = 8;
	const int ldsBlockdim = 10;
	const int ldsAccessOffset = 1;
	const int kernelDim = 3;

	__shared__ LDS sharedBuffer[ldsBlockdim * ldsBlockdim];

	// calculate address
	int x = threadIdx.x + blockIdx.x * blockdim;
	int y = threadIdx.y + blockIdx.y * blockdim;

	int id = (threadIdx.x + threadIdx.y * blockdim);

	int x1 = blockIdx.x * blockdim - ldsAccessOffset + id % ldsBlockdim;
	int y1 = blockIdx.y * blockdim - ldsAccessOffset + id / ldsBlockdim;

	int x2 = blockIdx.x * blockdim - ldsAccessOffset + (id + blockdim * blockdim) % ldsBlockdim;
	int y2 = blockIdx.y * blockdim - ldsAccessOffset + (id + blockdim * blockdim) / ldsBlockdim;

	// global load 1
	Float3Ushort1 colorAndMask1 = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1));
	Float2 normalAndDepth1      = Load2DFloat2(normalDepthBuffer, Int2(x1, y1));

	Float3 colorValue1          = colorAndMask1.xyz;
	float depthValue1           = normalAndDepth1.y;
	Float3 normalValue1         = DecodeNormal_R11_G10_B11(normalAndDepth1.x);
	ushort maskValue1           = colorAndMask1.w;

	// store to lds 1
	sharedBuffer[id      ] = { colorValue1, depthValue1, normalValue1, maskValue1 };

	if (id + blockdim * blockdim < ldsBlockdim * ldsBlockdim)
	{
		// global load 2
		Float3Ushort1 colorAndMask2 = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2));
		Float2 normalAndDepth2 = Load2DFloat2(normalDepthBuffer, Int2(x2, y2));

		Float3 colorValue2 = colorAndMask2.xyz;
		float depthValue2 = normalAndDepth2.y;
		Float3 normalValue2 = DecodeNormal_R11_G10_B11(normalAndDepth2.x);
		ushort maskValue2 = colorAndMask2.w;

		// store to lds 2
		sharedBuffer[id + blockdim * blockdim] = { colorValue2, depthValue2, normalValue2, maskValue2 };
	}

	__syncthreads();

	if (x >= size.x || y >= size.y) return;

	// load center
	LDS center   = sharedBuffer[threadIdx.x + ldsAccessOffset + (threadIdx.y + ldsAccessOffset) * ldsBlockdim];
	Float3 colorValue  = center.color;
	float depthValue   = center.depth;
	Float3 normalValue = center.normal;
	ushort maskValue   = center.mask;

	if (depthValue >= RayMax) { return; }

	// --------------------------------  filter --------------------------------
	LumaIdxPair neighbourLumaIdx[9];
	#pragma unroll
	for (int i = 0; i < kernelDim; ++i)
	{
		#pragma unroll
		for (int j = 0; j < kernelDim; ++j)
		{
			LDS bufferReadTmp = sharedBuffer[threadIdx.x + j + (threadIdx.y + i) * ldsBlockdim];
			Float3 color  = bufferReadTmp.color;
			float depth   = bufferReadTmp.depth;
			Float3 normal = bufferReadTmp.normal;
			ushort mask   = bufferReadTmp.mask;

			neighbourLumaIdx[j + i * kernelDim].luma = GetLuma(color);
			neighbourLumaIdx[j + i * kernelDim].idx = j + i * kernelDim;
		}
	}

	// sort neighbour
	sort9(neighbourLumaIdx);

	// get idx
	int medianColorIdx = neighbourLumaIdx[4].idx;

	// get values
	Float3 medianColor  = sharedBuffer[medianColorIdx].color;
	float  medianDepth  = sharedBuffer[medianColorIdx].depth;
	Float3 medianNormal = sharedBuffer[medianColorIdx].normal;
	ushort medianMask   = sharedBuffer[medianColorIdx].mask;

	// calculate blend factor
	Float3 t;
	float dist2;
	float weight = 1.0f;

	// normal diff factor
	t            = normalValue - medianNormal;
	dist2        = dot(t,t);
	weight       *= min1f(expf(-(dist2) / 0.1f), 1.0f);

	// depth diff fatcor
	dist2        = depthValue - medianDepth;
	dist2        = dist2 * dist2;
	weight      *= min1f(expf(-(dist2) / 0.1f), 1.0f);

	// material mask diff factor
	dist2        = (maskValue != medianMask) ? 1.0f : 0.0f;
	weight      *= min1f(expf(-(dist2) / 0.1f), 1.0f);

	// final blend
	Float3 finalColor = lerp3f(medianColor, lerp3f(colorValue, medianColor, weight), 0.5f);

	// handles nan
	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        printf("MedianFilter: nan found at (%d, %d)\n", x, y);
        finalColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { finalColor, maskValue } , colorBuffer, Int2(x, y));
}



//----------------------------------------------------------------------------------------------
//------------------------------------- Bloom ---------------------------------------
//----------------------------------------------------------------------------------------------

__global__ void BloomGuassian(SurfObj outBuffer, SurfObj inBuffer, Int2 size, float* exposure)
{
	// index for pixel 28 x 28
	Int2 idx2;
	idx2.x = blockIdx.x * 12 + threadIdx.x - 2;
	idx2.y = blockIdx.y * 12 + threadIdx.y - 2;

	// read global memory buffer. One-to-one mapping
	Float3 inColor = Load2DHalf4(inBuffer, idx2).xyz;

	// discard dark pixels
	float lum = inColor.getmax();
	float brightLum = exposure[2];

	inColor *= max(sqrtf(lum * lum - brightLum), 0.0);

	// Save global memory to shared memory. One-to-one mapping
	__shared__ Float3 sharedBuffer[16][16];

	sharedBuffer[threadIdx.x][threadIdx.y] = inColor;
	__syncthreads();

	// Border margin pixel finish work
	if (threadIdx.x < 2 || threadIdx.y < 2 || threadIdx.x > 13 || threadIdx.y > 13) { return; }

	Float3 outColor = 0;
	float weight = 0;

	#pragma unroll
	for (int i = 0; i < 25; ++i)
	{
		int x = threadIdx.x + (i % 5) - 2;
		int y = threadIdx.y + (i / 5) - 2;
		outColor += GetGaussian5x5(i) * sharedBuffer[x][y];
		weight += GetGaussian5x5(i);
	}

	outColor /= weight;

	NAN_DETECTER(outColor);

	Store2DHalf4(Float4(outColor, 1.0), outBuffer, idx2);
}

__global__ void Bloom(SurfObj colorBuffer, SurfObj buffer4, SurfObj buffer16, Int2 size, Int2 size4, Int2 size16)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx.x >= size.x || idx.y >= size.y) return;
	Float2 uv((float)idx.x / size.x, (float)idx.y / size.y);

	Float3 sampledColor4 = SampleBicubicCatmullRom(buffer4, Load2DHalf4ToFloat3, uv, size4);
	Float3 sampledColor16 = SampleBicubicCatmullRom(buffer16, Load2DHalf4ToFloat3, uv, size16);
	Float3 color = Load2DHalf4(colorBuffer, idx).xyz;

	color = color + (sampledColor4 + sampledColor16) * 0.05;

	NAN_DETECTER(color);

	Store2DHalf4(Float4(color, 1.0), colorBuffer, idx);
}

//----------------------------------------------------------------------------------------------
//------------------------------------- Lens Flare ---------------------------------------
//----------------------------------------------------------------------------------------------

__device__ __inline__ float LensFlareRand(float w)
{
    return fract(sinf(w) * 1000.0f);
}

__device__ __inline__ float LensFlareRegShape(Float2 p, int N)
{
	float a = atan2f(p.x, p.y) + 0.2f;
	float b = TWO_PI / float(N);
	float f = smoothstep1f(0.5f, 0.51f, cosf(floorf(0.5f + a / b) * b - a) * p.length());
    return f;
}

__device__ __inline__ Float3 LensFlareCircle(Float2 p, float size, float dist, Float2 mouse)
{
	float l = length(p + mouse * (dist * 4.0)) + size / 2.0f;

	float c  = max(0.01f - powf(length(p + mouse * dist), size * 1.4f), 0.0f) * 30.0f; // big circle
	float c1 = max(0.001f - powf(l - 0.3f, 1.0f / 40.0f) + sinf(l * 30.0f), 0.0f) * 3.0f; // ring
	float c2 = max(0.04f / powf(length(p - mouse * dist / 2.0f + 0.09f) * 1.0f, 1.0f), 0.0f) / 20.0f; // dot
	float s  = max(0.01f - powf(LensFlareRegShape(p * 5.0f + mouse * dist * 5.0f + 0.9f, 6.0f), 1.0f), 0.0f) * 6.0f; // Hexagon

	Float3 color = cos3f(Float3(0.44f, 0.24f, 0.2f) * 8.0f + dist * 4.0f) * 0.5f + 0.5f;

	Float3 f = c * color;
	f += c1 * color;
	f += c2 * color;
	f += s * color;

	return f - 0.01f;
}

__global__ void LensFlare(Float2 sunPos, SurfObj colorBuffer, Int2 size)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx.x >= size.x || idx.y >= size.y) return;

	Float2 uv((float)idx.x / (float)size.x, (float)idx.y / (float)size.y);
	uv -= Float2(0.5f);
	uv.x *= (float)size.x / (float)size.y;

	Float2 vec = uv - sunPos;
	float len = vec.length();

	Float3 color = Load2DHalf4(colorBuffer, idx).xyz;

	#pragma unroll
	for (int i = 0; i < 2; ++i)
	{
        color += LensFlareCircle(uv,
			                     powf(LensFlareRand(i * 2000.0f) * 1.8f, 2.0f) + 1.41f,
			                     LensFlareRand(i * 20.0f) * 3.0f + 0.2f - 0.5f,
			                     sunPos);
    }

	// get angle and length of the sun (uv - mouse)
    float angle = atan2f(vec.y, vec.x);

    // add the sun with the frill things
    color += max(0.1f / max(powf(len * 10.0f, 5.0f), 0.0001f), 0.0f) * abs(sinf(angle * 5.0f + cosf(angle * 9.0f))) / 20.0f;
    color += max(0.1f / powf(len * 10.0f, 1.0f / 20.0f), 0.0f) + abs(sinf(angle * 3.0f + cosf(angle * 9.0f))) / 16.0f * abs(sinf(angle * 9.0f));

	Store2DHalf4(Float4(color, 1.0), colorBuffer, idx);
}

__global__ void LensFlarePred(SurfObj depthBuffer, Float2 sunPos, Int2 sunUv, SurfObj colorBufferA, Int2 size, dim3 gridDim, dim3 blockDim, Int2 bufferDim)
{
	float depthValue = Load2DHalf1(depthBuffer, sunUv);
	if (depthValue < RayMax) { return; }
	LensFlare <<<gridDim, blockDim>>> (sunPos, colorBufferA, bufferDim);
}

//----------------------------------------------------------------------------------------------
//------------------------------------- Tone mapping ---------------------------------------
//----------------------------------------------------------------------------------------------

__device__ __forceinline__ float luminance(Float3 v)
{
    return dot(v, Float3(0.2126f, 0.7152f, 0.0722f));
}

__device__ __forceinline__ Float3 ChangeLuminance(Float3 c_in, float l_out)
{
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

__device__ __forceinline__ Float3 Reinhard(Float3 v)
{
    return v / (1.0f + v);
}

__device__ __forceinline__ Float3 ReinhardExtended(Float3 v, float max_white)
{
    Float3 numerator = v * (1.0f + (v / Float3(max_white * max_white)));
    return numerator / (1.0f + v);
}

__device__ __forceinline__ Float3 ReinhardExtendedLuminance(Float3 v, float max_white_l)
{
    float l_old = luminance(v);
    float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    float l_new = numerator / (1.0f + l_old);
    return ChangeLuminance(v, l_new);
}

__global__ void ToneMappingReinhard(
	SurfObj   colorBuffer,
	Int2      size,
	float*    exposure,
	PostProcessParams params)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 color = Load2DHalf4(colorBuffer, idx).xyz;
	color *= exposure[0];

	// Tone mapping
	color = Reinhard(color);

	// Gamma correction
	color = clamp3f(pow3f(color, 1.0f / params.gamma), Float3(0), Float3(1));

	Store2DHalf4(Float4(color, 1.0), colorBuffer, idx);
}

__global__ void ToneMappingReinhardExtended(
	SurfObj   colorBuffer,
	Int2      size,
	float*    exposure,
	PostProcessParams params)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 color = Load2DHalf4(colorBuffer, idx).xyz;
	color *= exposure[0];

	// Tone mapping
	color = ReinhardExtendedLuminance(color, params.W);

	// Gamma correction
	color = clamp3f(pow3f(color, 1.0f / params.gamma), Float3(0), Float3(1));

	Store2DHalf4(Float4(color, 1.0), colorBuffer, idx);
}

__device__ __forceinline__ Float3 ACESFilm(Float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp3f((x*(a*x+b))/(x*(c*x+d)+e));
}

__global__ void ToneMappingACES2(
	SurfObj   colorBuffer,
	Int2      size,
	float*    exposure,
	PostProcessParams params)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 color = Load2DHalf4(colorBuffer, idx).xyz;
	color *= exposure[0];

	// Tone mapping
	color = ACESFilm(color);

	// Gamma correction
	color = clamp3f(pow3f(color, 1.0f / params.gamma), Float3(0), Float3(1));

	Store2DHalf4(Float4(color, 1.0), colorBuffer, idx);
}

__device__ __forceinline__ Float3 RRTAndODTFit(Float3 v)
{
    Float3 a = v * (v + 0.0245786f) - 0.000090537f;
    Float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

__device__ __forceinline__ Float3 RRTAndODTFitLuminance(Float3 v)
{
	float lum = luminance(v);
    float a = lum * (lum + 0.0245786f) - 0.000090537f;
    float b = lum * (0.983729f * lum + 0.4329510f) + 0.238081f;
    return ChangeLuminance(v, a / b);
}

__device__ __forceinline__ Float3 ACESFitted(Float3 color)
{
	// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
	Mat3 ACESInputMat(
		0.59719, 0.35458, 0.04823,
		0.07600, 0.90834, 0.01566,
		0.02840, 0.13383, 0.83777);

	// ODT_SAT => XYZ => D60_2_D65 => sRGB
	Mat3 ACESOutputMat(
		1.60475, -0.53108, -0.07367,
		-0.10208,  1.10813, -0.00605,
		-0.00327, -0.07276,  1.07602);

    color = ACESInputMat * color;

    // Apply RRT and ODT
	// color = RRTAndODTFit(color);
    color = RRTAndODTFitLuminance(color);

    color = ACESOutputMat * color;

    // Clamp to [0, 1]
    color = clamp3f(color);

    return color;
}

__global__ void ToneMappingACES(
	SurfObj   colorBuffer,
	Int2      size,
	float*    exposure,
	PostProcessParams params)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	Float3 color = Load2DHalf4(colorBuffer, idx).xyz;
	color *= exposure[0];

	// Tone mapping
	color = ACESFitted(color);

	// Gamma correction
	color = clamp3f(pow3f(color, 1.0f / params.gamma), Float3(0), Float3(1));

	Store2DHalf4(Float4(color, 1.0), colorBuffer, idx);
}

__device__ __forceinline__ Float3 Uncharted2Tonemap(Float3 x, PostProcessParams params)
{
	const float A = params.A;
	const float B = params.B;
	const float C = params.C;
	const float D = params.D;
	const float E = params.E;
	const float F = params.F;

	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

__global__ void ToneMappingUncharted(
	SurfObj   colorBuffer,
	Int2      size,
	float*    exposure,
	PostProcessParams params)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= size.x || idx.y >= size.y) return;

	const float W = params.W;

	Float3 texColor = Load2DHalf4(colorBuffer, idx).xyz;
	texColor *= exposure[0];

	float ExposureBias = 2.0f;
	Float3 curr = Uncharted2Tonemap(ExposureBias * texColor, params);

	Float3 whiteScale = 1.0f/Uncharted2Tonemap(W, params);
	Float3 color = curr * whiteScale;

	// Gamma correction
	Float3 retColor = clamp3f(pow3f(color, 1.0f / params.gamma), Float3(0), Float3(1));

	Store2DHalf4(Float4(retColor, 1.0), colorBuffer, idx);
}

//----------------------------------------------------------------------------------------------
//------------------------------------- Scale Filter to Output ---------------------------------------
//----------------------------------------------------------------------------------------------

__device__ __inline__ Float3 min3f3(const Float3& v1, const Float3& v2, const Float3& v3) { return min3f(min3f(v1, v2), v3); }
__device__ __inline__ Float3 max3f3(const Float3& v1, const Float3& v2, const Float3& v3) { return max3f(max3f(v1, v2), v3); }

__device__ __forceinline__ Float3 SoftMin(Float3 **a, int x, int y)
{
	Float3 tmp1 = min3f3(a[x][y], a[x - 1][y], a[x + 1][y]);
	Float3 tmp2 = min3f3(tmp1, a[x][y - 1], a[x][y + 1]);
	Float3 tmp3 = min3f3(tmp2, a[x - 1][y - 1], a[x - 1][y + 1]);
	Float3 tmp4 = min3f3(tmp3, a[x + 1][y - 1], a[x + 1][y + 1]);
	return (tmp2 + tmp4);
}

__global__ void SharpeningFilter(SurfObj colorBuffer, Int2 texSize)
{
	// Reference: https://github.com/GPUOpen-Effects/FidelityFX-CAS

	const float sharpness = 1.0f;

	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= texSize.x || idx.y >= texSize.y) return;

	Float3 color[3][3];

	#pragma unroll
	for (int i = 0; i <= 2; ++i)
	{
		#pragma unroll
		for (int j = 0; j <= 2; ++j)
		{
			color[i][j] = Load2DHalf4(colorBuffer, idx + Int2(i - 1, j - 1)).xyz;
		}
	}

	// ----------------soft max------------------
	//        a b c             b
	// (max ( d e f ) + max ( d e f )) * 0.5
	//        g h i             h
	int x = 1, y = 1;
	Float3 tmp1 = max3f3(color[x][y], color[x - 1][y], color[x + 1][y]);
	Float3 tmp2 = max3f3(tmp1, color[x][y - 1], color[x][y + 1]);
	Float3 tmp3 = max3f3(tmp2, color[x - 1][y - 1], color[x - 1][y + 1]);
	Float3 tmp4 = max3f3(tmp3, color[x + 1][y - 1], color[x + 1][y + 1]);
	Float3 softmax = tmp2 + tmp4;

	// soft min
	tmp1 = min3f3(color[x][y], color[x - 1][y], color[x + 1][y]);
	tmp2 = min3f3(tmp1, color[x][y - 1], color[x][y + 1]);
	tmp3 = min3f3(tmp2, color[x - 1][y - 1], color[x - 1][y + 1]);
	tmp4 = min3f3(tmp3, color[x + 1][y - 1], color[x + 1][y + 1]);
	Float3 softmin = tmp2 + tmp4;

	// amp factor
	Float3 amp = clamp3f(min3f(softmin, 2.0f - softmax) / softmax);
	amp = rsqrt3f(amp);

	// weight
	float peak = 8.0f - 3.0f * sharpness;
    Float3 w = - Float3(1.0f) / (amp * peak);

	// 0 w 0
	// w 1 w
	// 0 w 0
	Float3 outColor = (color[0][1] + color[2][1] + color[1][0] + color[1][2]) * w + color[1][1];
	outColor /= (Float3(1.0f) + Float3(4.0f) * w);

	Store2DHalf4(Float4(outColor, 1.0f), colorBuffer, idx);
}

__global__ void BicubicScale(
	SurfObj  outputResColor,
	SurfObj  finalColorBuffer,
	Int2     outSize,
	Int2     texSize)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= outSize.x || idx.y >= outSize.y) return;

	Float2 uv((float)idx.x / outSize.x, (float)idx.y / outSize.y);

	Float3 sampledColor = SampleBicubicCatmullRom(finalColorBuffer, Load2DHalf4ToFloat3, uv, texSize);

	Store2DHalf4(Float4(sampledColor, 1.0f), outputResColor, idx);
}