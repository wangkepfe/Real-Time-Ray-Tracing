#pragma once

#include <cuda_runtime.h>
#include "debugUtil.h"
#include "sampler.cuh"
#include "settingParams.h"
#include "skyData.h"

inline __device__ Float3 EqualRectMap(float u, float v)
{
	float theta = u * TWO_PI;
	float phi = v * PI_OVER_2;

	float x = cos(theta) * cos(phi);
	float y = sin(phi);
	float z = sin(theta) * cos(phi);

	return Float3(x, y, z);
}

inline __device__ Float2 EqualRectMap(Float3 dir)
{
	float phi = asin(dir.y);
	float theta = acos(dir.x / cos(phi));

	float u = theta / TWO_PI;
	float v = phi / PI_OVER_2;

	return Float2 (u, v);
}

inline __device__ Float3 EqualAreaMap(float u, float v)
{
	float z = v;
	float r = sqrtf(1.0f - v * v);
	float phi = TWO_PI * u;

	return Float3 (r * cosf(phi), z, r * sinf(phi));
}

 __device__ Float2 EqualAreaMap(Float3 dir)
{
	float u = atan2f(-dir.z, -dir.x) / TWO_PI + 0.5f;
	float v = max(dir.y, 0.0f);
	return Float2 (u, v);
}

inline __device__ Float3 EqualAreaMapCone(const Float3& sunDir, float u, float v, float cosThetaMax)
{
    float cosTheta = (1.0f - u) + u * cosThetaMax;
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float phi = v * TWO_PI;

	Float3 t, b;
	LocalizeSample(sunDir, t, b);
	Mat3 trans(t, sunDir, b);

	Float3 coords = Float3(cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);

    return trans * coords;
}

inline __device__ bool EqualAreaMapCone(Float2& uv, const Float3& sunDir, const Float3& rayDir, float cosThetaMax)
{
	Float3 t, b;
	LocalizeSample(sunDir, t, b);
	Mat3 trans(t, sunDir, b);
	trans.transpose();

	Float3 coords = trans * rayDir;
	float cosTheta = coords.y;
	float u = (1.0f - cosTheta) / (1.0f - cosThetaMax);

	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	if (sinTheta < 1e-5f) {
		return false;
	}

	float cosPhi = coords.x / sinTheta;
	if (cosPhi < -1.0f || cosPhi > 1.0f) {
		return false;
	}

	float v = acosf(cosPhi) * INV_TWO_PI;

	uv = Float2(u, v);

	return true;
}

//(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
inline float GetFittingData(const float* elev_matrix, float solar_elevation, int i)
{
	return ( powf(1.0f-solar_elevation, 5.0f) * elev_matrix[i]  +
				5.0f  * powf(1.0f-solar_elevation, 4.0f) * solar_elevation * elev_matrix[i+9] +
				10.0f*powf(1.0f-solar_elevation, 3.0f)*powf(solar_elevation, 2.0f) * elev_matrix[i+18] +
				10.0f*powf(1.0f-solar_elevation, 2.0f)*powf(solar_elevation, 3.0f) * elev_matrix[i+27] +
				5.0f*(1.0f-solar_elevation)*powf(solar_elevation, 4.0f) * elev_matrix[i+36] +
				powf(solar_elevation, 5.0f)  * elev_matrix[i+45]);
}

//(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
inline float GetFittingData2(const float* elev_matrix, float solar_elevation)
{
	return ( powf(1.0f-solar_elevation, 5.0f) * elev_matrix[0] +
				5.0f*powf(1.0f-solar_elevation, 4.0f)*solar_elevation * elev_matrix[1] +
				10.0f*powf(1.0f-solar_elevation, 3.0f)*powf(solar_elevation, 2.0f) * elev_matrix[2] +
				10.0f*powf(1.0f-solar_elevation, 2.0f)*powf(solar_elevation, 3.0f) * elev_matrix[3] +
				5.0f*(1.0f-solar_elevation)*powf(solar_elevation, 4.0f) * elev_matrix[4] +
				powf(solar_elevation, 5.0f) * elev_matrix[5]);
}


__constant__ float skyConfigs[90];
__constant__ float skyRadiances[10];
// __constant__ float solarDatasets[1800];
// __constant__ float limbDarkeningDatasets[60];

// inline void InitSkyConstantBuffer()
// {
// 	GpuErrorCheck(cudaMemcpyToSymbol(solarDatasets, h_solarDatasets, sizeof(float) * 1800));
// 	GpuErrorCheck(cudaMemcpyToSymbol(limbDarkeningDatasets, h_limbDarkeningDatasets, sizeof(float) * 60));
// }

inline void UpdateSkyState(const Float3& sunDir)
{
	float h_skyConfigs[90];
	float h_skyRadiances[10];

	float elevation = acos(sunDir.y);

	float solar_elevation = powf(elevation / (M_PI / 2.0f), (1.0f / 3.0f));

	unsigned int channel;
    for( channel = 0; channel < 10; ++channel )
    {
		for(int i = 0; i < 9; ++i)
		{
			h_skyConfigs[channel * 9 + i] = GetFittingData(skyDataSets + channel * 54, solar_elevation, i);
		}

		h_skyRadiances[channel] = GetFittingData2(skyDataSetsRad + channel * 6, solar_elevation);
	}

	GpuErrorCheck(cudaMemcpyToSymbol(skyConfigs, h_skyConfigs, sizeof(float) * 10 * 9));
	GpuErrorCheck(cudaMemcpyToSymbol(skyRadiances, h_skyRadiances, sizeof(float) * 10));
}

inline __device__ Float3 SpectrumToXyz(uint channel)
{
	static const float spectrumCieX[] = {
		2.372527e-02f, 1.955480e+00f, 1.074553e+01f, 5.056697e+00f, 4.698190e+00f, 2.391135e+01f, 3.798705e+01f, 1.929414e+01f, 2.970610e+00f, 2.092986e-01f,
	};

	static const float spectrumCieY[] = {
		6.813859e-04f, 6.771017e-02f, 1.171193e+00f, 6.997765e+00f, 2.666710e+01f, 3.758372e+01f, 2.503930e+01f, 8.150395e+00f, 1.098635e+00f, 7.563256e-02f,
	};

	static const float spectrumCieZ[] = {
		1.119121e-01f, 9.441195e+00f, 5.597921e+01f, 3.589996e+01f, 5.070894e+00f, 3.523189e-01f, 3.422707e-02f, 2.539118e-03f, 7.836666e-06f, 0.000000e+00f,
	};

	return Float3(spectrumCieX[channel], spectrumCieY[channel], spectrumCieZ[channel]);
}

inline __device__ Float3 XyzToRgb(Float3 xyzColor)
{
	// ACES 2065-1 D60
	const Mat3 xyzToRgb(
		1.0498110175f , 0.0f        , -0.0000974845f ,
		-0.4959030231f, 1.3733130458f, 0.0982400361f   ,
		0.0f          , 0.0f        , 0.9912520182f);

	// SRGB D65
	// static const Mat3 xyzToRgb(
	// 	3.2404542, -1.5371385, -0.4985314,
	// 	-0.9692660,  1.8760108,  0.0415560,
	// 	0.0556434, -0.2040259,  1.0572252);

	Float3 rgbColor = xyzToRgb * xyzColor;

	return rgbColor;
}

inline __device__ Float3 GetSkyRadiance(const Float3& raydir, const Float3& sunDir, SkyParams& skyParams)
{
	float theta = acos(raydir.y);
	float gamma = acos(clampf(dot(raydir, sunDir), -1, 1));

    unsigned int channel;

	float spectrum[10];
	Float3 xyzColor = 0;

	#pragma unroll
	for( channel = 0; channel < 10; ++channel )
    {
		float* configuration = skyConfigs + channel * 9;

		const float expM = exp(configuration[4] * gamma);
		const float rayM = cos(gamma)*cos(gamma);
		const float mieM = (1.0f + cos(gamma)*cos(gamma)) / powf((1.0f + configuration[8]*configuration[8] - 2.0f*configuration[8]*cos(gamma)), 1.5);
		const float zenith = sqrt(cos(theta));

		float radianceInternal = (1.0f + configuration[0] * exp(configuration[1] / (cos(theta) + 0.01))) * (configuration[2] + configuration[3] * expM + configuration[5] * rayM + configuration[6] * mieM + configuration[7] * zenith);

		float radiance = radianceInternal * skyRadiances[channel];

		spectrum[channel] = radiance;

		xyzColor += radiance * SpectrumToXyz(channel);
	}

	Float3 rgbColor = XyzToRgb(xyzColor);

	return rgbColor;
}

inline __device__ Float3 GetSunRadiance(const Float3& raydir, const Float3& sunDir, SkyParams& skyParams) { return 0; }
// {
// 	float theta = acos(raydir.y);
// 	float gamma = acos(clampf(dot(raydir, sunDir), -1, 1));

//     unsigned int channel;

// 	float elevation = (M_PI / 2.0f) - acos(sunDir.y);

// 	constexpr float sunAreaScaleFactor = 4.0f;
// 	constexpr float sunBrightnessScaleFactor = 1.0f / (sunAreaScaleFactor * sunAreaScaleFactor);
// 	constexpr float solarRadius = 0.51f * M_PI / 180.0f / 2.0f * sunAreaScaleFactor;

// 	Float3 xyzColor = 0;

// 	float sol_rad_sin = sinf(solarRadius);
//     float ar2 = 1.0f / ( sol_rad_sin * sol_rad_sin );
//     float singamma = sin(gamma);
//     float sc2 = 1.0f - ar2 * singamma * singamma;

//     if (sc2 < 0.0f)
// 		sc2 = 0.0f;

// 	float sampleCosine = sqrt (sc2);

// 	if (sampleCosine == 0.0f)
// 		return 0.0f;

// 	#pragma unroll
// 	for( channel = 0; channel < 10; ++channel )
// 	{
// 		const int pieces = 45;
// 		const int order = 4;

// 		int pos = (int) (powf(2.0*elevation / M_PI, 1.0/3.0) * pieces); // floor

// 		if ( pos > 44 )
// 			pos = 44;

// 		const float break_x = powf(((float) pos / (float) pieces), 3.0) * (M_PI * 0.5);

// 		const float* coefs = solarDatasets + channel * 180 + (order * (pos+1) - 1);

// 		float res = 0.0;
// 		const float x = elevation - break_x;
// 		float x_exp = 1.0;

// 		int i;
// 		#pragma unroll
// 		for (i = 0; i < order; ++i)
// 		{
// 			res += x_exp * *coefs--;
// 			x_exp *= x;
// 		}

// 		float directRadiance = res;

// 		float ldCoefficient[6];

// 		#pragma unroll
// 		for ( i = 0; i < 6; i++ )
// 			ldCoefficient[i] = limbDarkeningDatasets[channel * 6 + i];

// 		float darkeningFactor =
// 			ldCoefficient[0]
// 			+ ldCoefficient[1] * sampleCosine
// 			+ ldCoefficient[2] * powf( sampleCosine, 2.0f )
// 			+ ldCoefficient[3] * powf( sampleCosine, 3.0f )
// 			+ ldCoefficient[4] * powf( sampleCosine, 4.0f )
// 			+ ldCoefficient[5] * powf( sampleCosine, 5.0f );

// 		directRadiance *= darkeningFactor * sunBrightnessScaleFactor * skyParams.sunScalar;

// 		xyzColor += directRadiance * SpectrumToXyz(channel);
// 	}

// 	Float3 rgbColor = XyzToRgb(xyzColor);

// 	return rgbColor;
// }

__global__ void Sky(SurfObj skyBuffer, float* skyPdf, Int2 size, Float3 sunDir, SkyParams skyParams)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = ((float)x + 0.5f) / size.x;
	float v = ((float)y + 0.5f) / size.y;

	Float3 rayDir = EqualAreaMap(u, v);

	Float3 color = GetSkyRadiance(rayDir, sunDir, skyParams) * skyParams.skyScalar;

	color = max3f(color, Float3(0.0f));

	Store2D_float4(Float4(color, 0), skyBuffer, Int2(x, y));

	// sky cdf
	int i = size.x * y + x;
	skyPdf[i] = dot(color, Float3(0.3f, 0.6f, 0.1f));
}

__global__ void SkySun(SurfObj sunBuffer, float* sunPdf, Int2 size, Float3 sunDir, SkyParams skyParams)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = ((float)x + 0.5f) / size.x;
	float v = ((float)y + 0.5f) / size.y;

	Float3 raydir = EqualAreaMapCone(sunDir, u ,v, cos(skyParams.sunAngle * M_PI / 180.0f / 2.0f));

	Float3 color = GetSunRadiance(raydir, sunDir, skyParams) * skyParams.skyScalar;

	color = max3f(color, Float3(0.0f));

	Store2D_float4(Float4(color, 0), sunBuffer, Int2(x, y));

	// sky cdf
	int i = size.x * y + x;
	sunPdf[i] = dot(color, Float3(0.3f, 0.6f, 0.1f));
}