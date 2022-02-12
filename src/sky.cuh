#pragma once

#include <cuda_runtime.h>
#include "debugUtil.h"
#include "sampler.cuh"
#include "settingParams.h"
#include "skyData.cuh"

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

inline __device__ Float2 EqualAreaMap(Float3 dir)
{
	float u = atan2f(-dir.z, -dir.x) / TWO_PI + 0.5f;
	float v = max(dir.y, 0.0f);
	return Float2 (u, v);
}

inline __device__ Float3 EnvLight2(const Float3& raydir, float clockTime, bool isDiffuseRay, SurfObj skyBuffer, Float2 blueNoise)
{
	Float2 jitterSize = Float2(1.0f) / Float2(SKY_WIDTH, SKY_HEIGHT);
	Float2 jitter = (blueNoise * jitterSize - jitterSize * 0.5f);

	Float3 color = SampleBicubicSmoothStep(skyBuffer, Load2DFloat4ToFloat3ForSky, EqualAreaMap(raydir), Int2(SKY_WIDTH, SKY_HEIGHT));
	return color;
}

//(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
inline __device__ float GetFittingData(float* elev_matrix, float solar_elevation, int i)
{
	return ( powf(1.0f-solar_elevation, 5.0f) * elev_matrix[i]  +
				5.0f  * powf(1.0f-solar_elevation, 4.0f) * solar_elevation * elev_matrix[i+9] +
				10.0f*powf(1.0f-solar_elevation, 3.0f)*powf(solar_elevation, 2.0f) * elev_matrix[i+18] +
				10.0f*powf(1.0f-solar_elevation, 2.0f)*powf(solar_elevation, 3.0f) * elev_matrix[i+27] +
				5.0f*(1.0f-solar_elevation)*powf(solar_elevation, 4.0f) * elev_matrix[i+36] +
				powf(solar_elevation, 5.0f)  * elev_matrix[i+45]);
}

//(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
inline __device__ float GetFittingData2(float* elev_matrix, float solar_elevation)
{
	return ( powf(1.0f-solar_elevation, 5.0f) * elev_matrix[0] +
				5.0f*powf(1.0f-solar_elevation, 4.0f)*solar_elevation * elev_matrix[1] +
				10.0f*powf(1.0f-solar_elevation, 3.0f)*powf(solar_elevation, 2.0f) * elev_matrix[2] +
				10.0f*powf(1.0f-solar_elevation, 2.0f)*powf(solar_elevation, 3.0f) * elev_matrix[3] +
				5.0f*(1.0f-solar_elevation)*powf(solar_elevation, 4.0f) * elev_matrix[4] +
				powf(solar_elevation, 5.0f) * elev_matrix[5]);
}


inline __device__ Float3 GetEnvIncidentLight(const Float3& raydir, const Float3& sunDir, SkyParams& skyParams)
{
	Float3 result;

	struct ArHosekSkyModelState
	{
		float configs[3 * 9];
		float radiances[3];
	};

	ArHosekSkyModelState state;

    float solar_radius = 0.51f * 180.0f / M_PI / 2.0f;
    float elevation    = acos(sunDir.y);
	float theta = acos(raydir.y);
	float gamma = acos(clampf(dot(raydir, sunDir), -1, 1));

	float albedo = skyParams.groundAlbedo;

	float solar_elevation = powf(elevation / (M_PI / 2.0f), (1.0f / 3.0f));

    unsigned int channel;

	// preparation pass
	#pragma unroll
    for( channel = 0; channel < 3; ++channel )
    {
		#pragma unroll
		for(int i = 0; i < 9; ++i)
		{
			state.configs[channel * 9 + i] =
				GetFittingData(skyDataSets + channel * 54, solar_elevation, i) * (1.0f - albedo) +
				GetFittingData(skyDataSets + 3 * 54 + channel * 54, solar_elevation, i) * (albedo);
		}

		state.radiances[channel] =
			GetFittingData2(skyDataSetsRad + channel * 6, solar_elevation) * (1.0f - albedo) +
			GetFittingData2(skyDataSetsRad + 3 * 6 + channel * 6, solar_elevation) * (albedo);
	}

	// calculation pass
	#pragma unroll
	for( channel = 0; channel < 3; ++channel )
    {
		float* configuration = state.configs + channel * 9;

		const float expM = exp(configuration[4] * gamma);
		const float rayM = cos(gamma)*cos(gamma);
		const float mieM = (1.0f + cos(gamma)*cos(gamma)) / powf((1.0f + configuration[8]*configuration[8] - 2.0f*configuration[8]*cos(gamma)), 1.5);
		const float zenith = sqrt(cos(theta));

		float radianceInternal = (1.0f + configuration[0] * exp(configuration[1] / (cos(theta) + 0.01))) * (configuration[2] + configuration[3] * expM + configuration[5] * rayM + configuration[6] * mieM + configuration[7] * zenith);

		float radiance = radianceInternal * state.radiances[channel];

		result[channel] = radiance;
	}

#if SKY_CIE_XYZ_COLOR_SPACE
	// SRGB+D65
	// Mat3 xyzToRgb(
	// 	3.2404542f, -1.5371385f, -0.4985314f,
	// 	-0.9692660f,  1.8760108f,  0.0415560f,
	// 	0.0556434f, -0.2040259f,  1.0572252f);

	// ACES 2065-1 D60
	Mat3 xyzToRgb(
		1.0498110175f , 0.0f        , -0.0000974845f ,
		-0.4959030231f, 1.3733130458, 0.0982400361   ,
		0.0f          , 0.0f        , 0.9912520182f);

	result = xyzToRgb * result;
#endif

	return result;
}

__global__ void Sky(SurfObj skyBuffer, float* skyPdf, Int2 size, Float3 sunDir, SkyParams skyParams)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = ((float)x + 0.5f) / size.x;
	float v = ((float)y + 0.5f) / size.y;

	Float3 rayDir = EqualAreaMap(u, v);

	Float3 color = GetEnvIncidentLight(rayDir, sunDir, skyParams) * skyParams.skyScalar;

	color = max3f(color, Float3(0.0f));

	Store2D_float4(Float4(color, 0), skyBuffer, Int2(x, y));

	// sky cdf
	int i = size.x * y + x;
	skyPdf[i] = dot(color, Float3(0.3f, 0.6f, 0.1f));
}
