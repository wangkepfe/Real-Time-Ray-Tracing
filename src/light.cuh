#pragma once

#include "kernel.cuh"
#include "debugUtil.h"
#include "bsdf.cuh"

#define TOTAL_LIGHT_MAX_COUNT 8

template<typename T>
inline __device__ T BinarySearch(const T* array, int left, int right, T target)
{
    int mid;

    while (right - left > 1)
    {
        mid = (left + right) / 2;
        float midVal = array[mid];

        if (midVal < target)
        {
            left = mid;
        }
        else
        {
            right = mid;
        }
    }
    mid = left;

    return mid;
}

inline __device__ Float3 EnvLight2(
    const Float3& sunDir,
    const Float3& raydir,
    float clockTime,
    bool isDiffuseRay,
    SurfObj skyBuffer,
    SurfObj sunBuffer,
    float sunAngleCosThetaMax,
    Float2 blueNoise)
{
    Float3 color = 0;

    // Sky
    {
        // Float2 jitterSize = Float2(1.0f) / Float2(SKY_WIDTH, SKY_HEIGHT);
        // Float2 jitter = (blueNoise * jitterSize - jitterSize * 0.5f);
        color += SampleBicubicSmoothStep(skyBuffer, Load2DFloat4ToFloat3ForSky, EqualAreaMap(raydir), Int2(SKY_WIDTH, SKY_HEIGHT));
    }

    // Sun
    {
        // Float2 jitterSize = Float2(1.0f) / Float2(SUN_WIDTH, SUN_HEIGHT);
        // Float2 jitter = (blueNoise * jitterSize - jitterSize * 0.5f);

        Float2 uv;
        if (EqualAreaMapCone(uv, sunDir, raydir, sunAngleCosThetaMax))
        {
            Float3 sunColor = SampleBicubicSmoothStep(sunBuffer, Load2DFloat4ToFloat3, uv, Int2(SUN_WIDTH, SUN_HEIGHT));
            color += sunColor;
        }
    }

	return color;
}

__device__ __inline__ void SampleLight(
    ConstBuffer&            cbo,
    RayState&               rayState,
    SceneMaterial           sceneMaterial,
    Float3&                 lightSampleDir,
    float&                  lightSamplePdf,
    float&                  isDeltaLight,
    float*                  skyCdf,
    float*                  sunCdf,
    int&                    lightIdx,
    Float4&                 randNum)
{
#if RENDERI_SPHERE_LIGHT
	const int numSphereLight = sceneMaterial.numSphereLights;
	Sphere* sphereLights = sceneMaterial.sphereLights;
	const int sunLightIdx = numSphereLight;
	const int envLightIdx = numSphereLight + 1;
	for (; i < numSphereLight; ++i)
	{
#if 0
		Float3 vec = sphereLights[i].center - rayState.pos;
		float dist2 = vec.length2();
		if (dot(normal, vec) > 0/* && dist2 < 10.0f*/)
		{
			indexRemap[idx++] = i; // sphere light
		}
#else
		indexRemap[idx++] = i;
#endif
	}
#else
    // const int envLightIdx = 0;
	//const int sunLightIdx = 0;
#endif

    const Float3& normal = rayState.normal;

	float lightChoosePdf = 1.0f;
	// int sampledIdx;

	// int indexRemap[TOTAL_LIGHT_MAX_COUNT] = {};
	// int i = 0;
	// int idx = 0;

    // indexRemap[idx++] = i++; // env light
    //indexRemap[idx++] = i++; // sun/moon light

	// choose light
    // float chooseLightRand = randNum[0];
	// int sampledValue = (int)floorf(chooseLightRand * idx);
	// sampledIdx = indexRemap[sampledValue];
	// lightChoosePdf = 1.0f / (float)idx;

	// sample
	// Float2 lightSampleRand2(randNum[1], randNum[2]);

    //DEBUG_PRINT(sampledIdx);

	// if (sampledIdx == sunLightIdx)
	// {
    //     Float3 moonDir = -cbo.sunDir;
    //     lightSampleDir = cbo.sunDir.y > 0.0f ? cbo.sunDir : moonDir;

    //     if (dot(normal, lightSampleDir) > 0)
    //     {
    //         isDeltaLight   = true;
    //         lightIdx = ENV_LIGHT_ID;
    //     }
    //     else
    //     {
    //         return false;
    //     }
	// }
    // else
    // if (sampledIdx == envLightIdx)
	{
        // The accumulated all sky luminance
        const float maxSkyCdf = skyCdf[SKY_SIZE - 1];

        // The accumulated all sun luminance
        const float maxSunCdf = sunCdf[SUN_SIZE - 1];

        // Sample sky or sun pdf
        const float sampleSkyVsSunPdf = maxSkyCdf / (maxSkyCdf + maxSunCdf);

        float chooseSampleSkyVsSun;

        if (cbo.sampleParams.sampleSkyVsSunUseFluxWeight)
        {
            chooseSampleSkyVsSun = sampleSkyVsSunPdf;
        }
        else
        {
            chooseSampleSkyVsSun = cbo.sampleParams.sampleSkyVsSun;
        }

        // Choose to sample sky
        if (chooseSampleSkyVsSun > randNum[1])
        {
            // Binary search in range 0 to size-2, since we want result+1 to be the index, we'll need to subtract result for calculating PDF
            const int sampledSkyIdx = BinarySearch(skyCdf, 0, SKY_SIZE - 2, randNum[0] * maxSkyCdf) + 1;

            // Subtract neighbor CDF to get PDF, divided by max CDF to get the probability
            float sampledSkyPdf = (skyCdf[sampledSkyIdx] - skyCdf[sampledSkyIdx - 1]) / maxSkyCdf;

            // Each tile has area 2Pi / resolution, pdf = 1/area = resolution / 2Pi
            sampledSkyPdf = sampledSkyPdf * SKY_SIZE / TWO_PI;

            // Index to 2D coordinates
            float u = ((sampledSkyIdx % SKY_WIDTH) + 0.5f) / SKY_WIDTH;
            float v = ((sampledSkyIdx / SKY_WIDTH) + 0.5f) / SKY_HEIGHT;

            // Hemisphere projection
            Float3 rayDir = EqualAreaMap(u, v);

            // Set light sample direction and PDF
            lightSampleDir = rayDir;
            lightSamplePdf = sampledSkyPdf * lightChoosePdf * chooseSampleSkyVsSun;

            // Set light index for shadow ray rejection
            lightIdx = ENV_LIGHT_ID;
        }
        else // Choose to sample sun
        {
            // Binary search in range 0 to size-2, since we want result+1 to be the index, we'll need to subtract result for calculating PDF
            const int sampledSunIdx = BinarySearch(sunCdf, 0, SUN_SIZE - 2, randNum[0] * maxSunCdf) + 1;

            // Subtract neighbor CDF to get PDF, divided by max CDF to get the probability
            float sampledSunPdf = (sunCdf[sampledSunIdx] - sunCdf[sampledSunIdx - 1]) / maxSunCdf;

            // Each tile has area = coneAnglularArea / resolution, pdf = 1/area = resolution / (TWO_PI * (1.0f - cosThetaMax))
            sampledSunPdf = sampledSunPdf * SUN_SIZE / (TWO_PI * (1.0f - cbo.sunAngleCosThetaMax));

            // Index to 2D coordinates
            float u = ((sampledSunIdx % SUN_WIDTH) + 0.5f) / SUN_WIDTH;
            float v = ((sampledSunIdx / SUN_WIDTH) + 0.5f) / SUN_HEIGHT;

            // Hemisphere projection
            Float3 rayDir = EqualAreaMapCone(cbo.sunDir, u, v, cbo.sunAngleCosThetaMax);

            // Set light sample direction and PDF
            lightSampleDir = rayDir;
            lightSamplePdf = sampledSunPdf * lightChoosePdf * (1.0f - chooseSampleSkyVsSun);

            // Set light index for shadow ray rejection
            lightIdx = ENV_LIGHT_ID;

            // Debug
            // DEBUG_PRINT(maxSkyCdf);
            // DEBUG_PRINT(maxSunCdf);
            // DEBUG_PRINT(sampleSkyVsSunPdf);
            // DEBUG_PRINT(sampledSunIdx);
            // DEBUG_PRINT(sampledSunPdf);
            // DEBUG_PRINT(u);
            // DEBUG_PRINT(v);
            // DEBUG_PRINT(rayDir);
        }
	}
#if RENDER_SPHERE_LIGHT
	else
	{
		Sphere sphereLight = sphereLights[sampledIdx];

        // light vector / direction
		Float3 lightVec   = sphereLight.center - rayState.pos;
        float dist2       = lightVec.length2();

        Float3 lightDir   = normalize(lightVec);
        float radius2     = sphereLight.radius * sphereLight.radius;
        float cosThetaMax = sqrtf(max1f(dist2 - radius2, 0)) / sqrtf(dist2);

        // sample direction based on light direction and cosThetaMax
        Float3 u, v;
        LocalizeSample(lightDir, u, v);
        lightSampleDir = UniformSampleCone(lightSampleRand2, cosThetaMax, u, v, lightDir);
        lightSampleDir = normalize(lightSampleDir);

        // if direction in sight
        if (dot(normal, lightSampleDir) > 0.0)
        {
            // calculate pdf of the sample
            lightSamplePdf = UniformConePdf(cosThetaMax) * lightChoosePdf;
            lightIdx = sampledIdx;
        }
        else
        {
            return false;
        }
	}
#endif
    //DEBUG_PRINT(lightSampleDir);
}

__device__ inline Float3 GetLightSource(
    ConstBuffer& cbo,
    RayState& rayState,
    SceneMaterial sceneMaterial,
    SurfObj skyBuffer,
    SurfObj sunBuffer,
    Float4& randNum)
{
    // check for termination and hit light
    if (rayState.hitLight == false || rayState.isOccluded == true) { return 0; }

    Float3 lightDir = rayState.dir;
    Float3 L0 = Float3(0);

    // Different light source type
    if (rayState.matType == MAT_SKY)
    {
        Float3 envLightColor = EnvLight2(cbo.sunDir, lightDir, cbo.clockTime, rayState.isDiffuseRay, skyBuffer, sunBuffer, cbo.sunAngleCosThetaMax, Float2(randNum.x, randNum.y));
        L0 = envLightColor;
    }
    // else if (rayState.matType == EMISSIVE)
    // {
    //     // local light
    //     SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];
    //     L0 = mat.albedo;
    // }

    return L0;
}
