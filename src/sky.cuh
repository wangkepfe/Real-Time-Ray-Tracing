#pragma once

#include <cuda_runtime.h>
#include "geometry.cuh"
#include "debug_util.cuh"
#include "sampler.cuh"
#include "water.cuh"
#include "star.cuh"

#define USE_OCEAN 0
#define USE_STAR 0
#define USE_HALF_PRECISION_SKY 0

struct SkyParams
{
	// Observer
	float latitude;
	float longtitude;
	float elevation;
	float timeOfDay;
	float month;
	float day;

	// Earth
	float earthRadius = 6360.0e3f;
	float atmosphereRadius = 6420.0e3f;

	// Rayleigh (air)
	Float3 scatteringCoeffRayleigh;
	float atmosphereThickness;

	// Mie (aerosol)
	Float3 scatteringCoeffMie;
	Float3 absorptionCoeffMie;
	float g;
	float hazeThickness;
};

inline __device__ float RayleighPhaseFunc(float mu)
{
	return
			3.0f * (1.0f + mu*mu)
	/ //-----------------------------
				(16.0f * M_PI);
}

inline __device__ float HenyeyGreensteinPhaseFunc(float mu, float g)
{
	return
						(1.0f - g*g)
	/ //-------------------------------------------------------
		((4.0f * M_PI) * pow(1.0f + g*g - 2.0f*g*mu, 1.5f));
}

inline __device__ float MiePhaseFunc(float mu, float g)
{
	return
				    (3.0f * (1.0f - g*g) * (1.0f + mu*mu))
	/ //----------------------------------------------------------------------
		(8.0f * M_PI * (2.0f + g*g) * pow(1.0f + g*g - 2.0f*g*mu, 1.5f));
}

inline __device__ Float3 GetEnvIncidentLight(const Float3& raydir, const Float3& sunDir)
{
	int numSamples = 16;
	int numSamplesLight = 8;

	// Earth and atmosphere radius in meter
	const float earthRadius = 6360.0e3f;
	const float atmosphereRadius = 6420.0e3f;

	// Elevation
	const float elevation = 0.0f;

	// Sun radiance
	const Float3 sunPower = Float3(20.0f);

	// Phase function anisotropic factor
    const float g = 0.76f;

    // Scattering coefficients at sea level (meter)
    const Float3 scatteringCoeffRayleigh = Float3(5.5e-6f, 13.0e-6f, 22.4e-6f);
    const Float3 scatteringCoeffMie = Float3(21.0e-6f);
	const Float3 absorptionCoeffMie = Float3(2.1e-6f);

    // Thickness of the atmosphere (meter)
    const float heightRayleigh = 8.0e3f;
    const float heightMie = 1.2e3f;

	// Distance between eye and the edge of atmosphere, which is the total path to trace
    Ray eyeRay;
    eyeRay.orig = Float3(0, earthRadius + elevation, 0);
    eyeRay.dir = raydir;

    Sphere atmosphereSphere;
    atmosphereSphere.center = 0;
    atmosphereSphere.radius = atmosphereRadius;

	Sphere earthSphere;
    earthSphere.center = 0;
    earthSphere.radius = earthRadius;

    float errorSphereRayIntersect;
	float t = SphereRayIntersect(atmosphereSphere, eyeRay, errorSphereRayIntersect);

	// Marching step
    float marchStep = t / float(numSamples + 1);

	// Optical depth = integral of extinction coefficient = sum()
    float opticalDepthRayleigh = 0;
	float opticalDepthMie = 0;

	Float3 sumR = Float3(0);
	Float3 sumM = Float3(0);
	float marchPos = 0;

    for (int i = 0; i < numSamples; i++)
    {
		Float3 s = eyeRay.orig + eyeRay.dir * (marchPos + 0.5f * marchStep);

		Ray lightRay;
        lightRay.orig = s;
        lightRay.dir = sunDir;

		// Current sample point height
		float height = s.length() - earthRadius;

		// Optical depth integral
		float expHeightDeltaRayleigh = expf(-height / heightRayleigh) * marchStep;
		float expHeightDeltaMie = expf(-height / heightMie) * marchStep;

		opticalDepthRayleigh += expHeightDeltaRayleigh;
		opticalDepthMie += expHeightDeltaMie;

		// Sun ray blocked by earth
		float t2 = SphereRayIntersect(earthSphere, lightRay, errorSphereRayIntersect);
		if (t2 != RayMax)
		{
			marchPos += marchStep;
			continue;
		}

		// Optical depth for light ray integral
		float opticalDepthLightRayleigh = 0;
		float opticalDepthLightMie = 0;

		float t1 = SphereRayIntersect(atmosphereSphere, lightRay, errorSphereRayIntersect);

		float sunLightMarchPos = 0;
		float sunLightMarchStep = t1 / float(numSamplesLight + 1);

		for (int i = 0; i < numSamplesLight; i++)
		{
			Float3 s1 = lightRay.orig + lightRay.dir * (sunLightMarchPos + 0.5f * sunLightMarchStep);
			float height1 = s1.length() - earthRadius;

			opticalDepthLightRayleigh += expf(-height1 / heightRayleigh) * sunLightMarchStep;
			opticalDepthLightMie += expf(-height1 / heightMie) * sunLightMarchStep;

			sunLightMarchPos += sunLightMarchStep;
		}

		Float3 attenuationCoeffRayleigh = scatteringCoeffRayleigh;
		Float3 attenuationCoeffMie = scatteringCoeffMie + absorptionCoeffMie;

		Float3 totalTransmittance = attenuationCoeffRayleigh * (opticalDepthRayleigh + opticalDepthLightRayleigh) + 
		                            attenuationCoeffMie * (opticalDepthMie + opticalDepthLightMie);

		Float3 attenuation = exp3f(-totalTransmittance);

		Float3 scatteringCoeffIntegralRayleigh = expHeightDeltaRayleigh * scatteringCoeffRayleigh;
		Float3 scatteringCoeffIntegralfMie = expHeightDeltaMie * scatteringCoeffMie;

		sumR += scatteringCoeffIntegralRayleigh * attenuation;
		sumM += scatteringCoeffIntegralfMie * attenuation;
		
		marchPos += marchStep;
	}

	// Evaluate phase functions
	float mu = dot(eyeRay.dir, sunDir);
    float phaseRayleigh = RayleighPhaseFunc(mu);
	float phaseMie = MiePhaseFunc(mu, g);

	Float3 result = sunPower * (sumR * phaseRayleigh + 
	                            sumM * phaseMie);

	return result;
}


inline __device__ Float3 GetEnvIncidentLight2(const Float3& raydir, const Float3& sunDir)
{
	int numSamples = 32;
	int numSamplesLight = 16;

	// Earth and atmosphere radius in meter
	const float earthRadius = 6360.0e3f;
	const float atmosphereRadius = 6420.0e3f;

	// Elevation
	const float elevation = 0.0f;

	// Sun radiance
	const Float3 sunPower = Float3(20.0f);

	// Phase function anisotropic factor
    const float g = 0.76f;

    // Scattering coefficients at sea level (meter)
    const Float3 scatteringCoeffRayleigh = Float3(5.5e-6f, 13.0e-6f, 22.4e-6f);
    const Float3 scatteringCoeffMie = Float3(21.0e-6f);
	const Float3 absorptionCoeffMie = Float3(2.1e-6f);

    // Thickness of the atmosphere (meter)
    const float heightRayleigh = 20.0e3f;
    const float heightMie = 2.0e3f;

	// Distance between eye and the edge of atmosphere, which is the total path to trace
    Ray eyeRay;
    eyeRay.orig = Float3(0, earthRadius + elevation, 0);
    eyeRay.dir = raydir;

    Sphere atmosphereSphere;
    atmosphereSphere.center = 0;
    atmosphereSphere.radius = atmosphereRadius;

	Sphere earthSphere;
    earthSphere.center = 0;
    earthSphere.radius = earthRadius;

    float errorSphereRayIntersect;
	float t = SphereRayIntersect(atmosphereSphere, eyeRay, errorSphereRayIntersect);

	// Marching step
    float marchStep = t / float(numSamples + 1);

	// Optical depth = integral of extinction coefficient = sum()
    float opticalDepthRayleigh = 0;
	float opticalDepthMie = 0;

	Float3 sumR = Float3(0);
	Float3 sumM = Float3(0);
	float marchPos = 0;

    for (int i = 0; i < numSamples; i++)
    {
		Float3 s = eyeRay.orig + eyeRay.dir * (marchPos + 0.5f * marchStep);

		Ray lightRay;
        lightRay.orig = s;
        lightRay.dir = sunDir;

		// Current sample point height
		float height = s.length() - earthRadius;

		// Optical depth integral
		float expHeightDeltaRayleigh = expf(-height / heightRayleigh) * marchStep;
		float expHeightDeltaMie = expf(-height / heightMie) * marchStep;

		opticalDepthRayleigh += expHeightDeltaRayleigh;
		opticalDepthMie += expHeightDeltaMie;

		// Sun ray blocked by earth
		float t2 = SphereRayIntersect(earthSphere, lightRay, errorSphereRayIntersect);
		if (t2 != RayMax)
		{
			marchPos += marchStep;
			continue;
		}

		// Optical depth for light ray integral
		float opticalDepthLightRayleigh = 0;
		float opticalDepthLightMie = 0;

		float t1 = SphereRayIntersect(atmosphereSphere, lightRay, errorSphereRayIntersect);

		float sunLightMarchPos = 0;
		float sunLightMarchStep = t1 / float(numSamplesLight + 1);

		for (int i = 0; i < numSamplesLight; i++)
		{
			Float3 s1 = lightRay.orig + lightRay.dir * (sunLightMarchPos + 0.5f * sunLightMarchStep);
			float height1 = s1.length() - earthRadius;

			opticalDepthLightRayleigh += expf(-height1 / heightRayleigh) * sunLightMarchStep;
			opticalDepthLightMie += expf(-height1 / heightMie) * sunLightMarchStep;

			sunLightMarchPos += sunLightMarchStep;
		}

		Float3 attenuationCoeffRayleigh = scatteringCoeffRayleigh;
		Float3 attenuationCoeffMie = scatteringCoeffMie + absorptionCoeffMie;

		Float3 totalTransmittance = attenuationCoeffRayleigh * (opticalDepthRayleigh + opticalDepthLightRayleigh) + 
		                            attenuationCoeffMie * (opticalDepthMie + opticalDepthLightMie);

		Float3 attenuation = exp3f(-totalTransmittance);

		Float3 scatteringCoeffIntegralRayleigh = expHeightDeltaRayleigh * scatteringCoeffRayleigh;
		Float3 scatteringCoeffIntegralfMie = expHeightDeltaMie * scatteringCoeffMie;

		sumR += scatteringCoeffIntegralRayleigh * attenuation;
		sumM += scatteringCoeffIntegralfMie * attenuation;
		
		marchPos += marchStep;
	}

	// Evaluate phase functions
	float mu = dot(eyeRay.dir, sunDir);
    float phaseRayleigh = RayleighPhaseFunc(mu);
	float phaseMie = MiePhaseFunc(mu, g);

	Float3 result = sunPower * (sumR * phaseRayleigh + 
	                            sumM * phaseMie);

	return result;
}


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
	float r = sqrtf(1 - v * v);
	float phi = TWO_PI * u;

	return Float3 (r * cosf(phi), z, r * sinf(phi));
}

inline __device__ Float2 EqualAreaMap(Float3 dir)
{
	float u = atan2f(-dir.z, -dir.x) / TWO_PI + 0.5f;
	float v = abs(dir.y);
	return Float2 (u, v);
}

inline __device__ Float3 EnvLight2(const Float3& raydir, float clockTime, bool isDiffuseRay, SurfObj skyBuffer, Float2 blueNoise)
{
	Float3 rayDirOrRefl = raydir;
	Float3 beta = Float3(1.0);

#if USE_OCEAN
	if (rayDirOrRefl.y < 0.01 && isDiffuseRay == false) { OceanShader(rayDirOrRefl, beta, clockTime * 0.7); }
#else
	if (rayDirOrRefl.y < 0) { rayDirOrRefl.y *= -1; }
#endif

	float u = atan2f(-rayDirOrRefl.z, -rayDirOrRefl.x) / TWO_PI + 0.5f;
	float v = abs(rayDirOrRefl.y);
	
#if 0 // tex read sky
	float4 texRead = tex2D<float4>(skyTex, u, v);
	Float3 color = Float3(texRead.x , texRead.y, texRead.z);
#endif

	Float2 jitterSize = Float2(0.5f) / Float2(SKY_WIDTH, SKY_HEIGHT);
	Float2 jitter = (blueNoise * jitterSize - jitterSize * 0.5f);

#if USE_HALF_PRECISION_SKY // half precision sky
	Float3 color = SampleBicubicSmoothStep(skyBuffer, Load2DHalf4ToFloat3ForSky, HemisphereUniformMap(rayDirOrRefl) + jitter, Int2(SKY_WIDTH, SKY_HEIGHT));
#else // full precision sky
	Float3 color = SampleBicubicSmoothStep(skyBuffer, Load2DFloat4ToFloat3ForSky, EqualAreaMap(rayDirOrRefl) + jitter, Int2(SKY_WIDTH, SKY_HEIGHT));
#endif

	return color * beta;
}

__global__ void Sky(SurfObj skyBuffer, float* skyCdf, Int2 size, Float3 sunDir)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = ((float)x + 0.5f) / size.x;
	float v = ((float)y + 0.5f) / size.y;

	// ray dir
	Float3 rayDir = EqualAreaMap(u, v);

	// get sky color
	const int numSkySample = 16;
	const int numSkyLightSample = 8;
	Float3 sunOrMoonDir = sunDir;
	Float3 color = 0;

	if (sunOrMoonDir.y > -sin(Pi_over_180 * 30.0f))
		color += GetEnvIncidentLight2(rayDir, sunOrMoonDir);

	// if (sunOrMoonDir.y < sin(Pi_over_180 * 30.0f))
	// 	color += GetEnvIncidentLight2(rayDir, -sunOrMoonDir) * 0.005f;

	// store
	#if USE_HALF_PRECISION_SKY
	Store2DHalf4(Float4(color, 0), skyBuffer, Int2(x, y));
	#else
	Store2D_float4(Float4(color, 0), skyBuffer, Int2(x, y));
	#endif

	// sky cdf
	int i = size.x * y + x;
	skyCdf[i] = color.getmax();
}
