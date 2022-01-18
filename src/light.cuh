#pragma once

#include "kernel.cuh"
#include "debug_util.cuh"
#include "bsdf.cuh"

#define TOTAL_LIGHT_MAX_COUNT 8

__device__ __inline__ bool SampleLight(
    ConstBuffer&            cbo,
    RayState&               rayState,
    SceneMaterial           sceneMaterial,
    Float3&                 lightSampleDir,
    float&                  lightSamplePdf,
    float&                  isDeltaLight,
    float*                  skyCdf,
    int&                    lightIdx)
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
	const int sunLightIdx = 0;
	const int envLightIdx = 1;
#endif

    const Float3& normal = rayState.normal;

	float lightChoosePdf;
	int sampledIdx;

	int indexRemap[TOTAL_LIGHT_MAX_COUNT] = {};
	int i = 0;
	int idx = 0;

	indexRemap[idx++] = i++; // sun/moon light
    indexRemap[idx++] = i++; // env light

	// choose light
    float chooseLightRand = rayState.rand.x;
	int sampledValue = (int)floorf(chooseLightRand * idx);
	sampledIdx = indexRemap[sampledValue];
	lightChoosePdf = 1.0f;// / (float)idx;

	// sample
	Float2 lightSampleRand2(rayState.rand.z, rayState.rand.w);

    //DEBUG_PRINT(sampledIdx);

	if (sampledIdx == sunLightIdx)
	{
        Float3 moonDir = -cbo.sunDir;
        lightSampleDir = cbo.sunDir.y > 0.0f ? cbo.sunDir : moonDir;

        if (dot(normal, lightSampleDir) > 0)
        {
            isDeltaLight   = true;
            lightIdx = ENV_LIGHT_ID;
        }
        else
        {
            return false;
        }
	}
    else if (sampledIdx == envLightIdx)
	{
        float maxSkyCdf = skyCdf[1023];
        float sampledSkyValue = lightSampleRand2[0] * maxSkyCdf;

        // DEBUG_PRINT(maxSkyCdf);
        // DEBUG_PRINT(sampledSkyValue);

        int left = 0;
        int right = 1022;
        int mid;
        for (int j = 0; j < 8; ++j)
        {
            mid = (left + right) / 2;
            float midVal = skyCdf[mid];

            // DEBUG_PRINT(j);
            // DEBUG_PRINT(mid);
            // DEBUG_PRINT(midVal);

            if (midVal < sampledSkyValue)
            {
                left = mid;
            }
            else
            {
                right = mid;
            }
        }
        mid = (left + right) / 2;

        int sampledSkyIdx = mid + 1;
        float sampledSkyPdf = (skyCdf[mid + 1] - skyCdf[mid]) / maxSkyCdf; // choose 1 from 1024 tiles
        sampledSkyPdf = sampledSkyPdf * 1024 / TWO_PI; // each tile has area 2Pi / 1024

        // DEBUG_PRINT(sampledSkyIdx);
        // DEBUG_PRINT(sampledSkyPdf);

        // index to 2D coordinates
        float u = ((sampledSkyIdx % 64) + 0.5f) / 64;
        float v = ((sampledSkyIdx / 64) + 0.5f) / 16;

        // (u);
        // DEBUG_PRINT(v);

        // hemisphere projection
        float z = v;
        float r = sqrtf(1 - v * v);
        float phi = TWO_PI * u;
        Float3 rayDir(r * cosf(phi), z, r * sinf(phi));

        // DEBUG_PRINT(rayDir);

        if (dot(rayDir, normal) > 0)
        {
            lightSampleDir = rayDir;
		    lightSamplePdf = sampledSkyPdf * lightChoosePdf;

            lightIdx = ENV_LIGHT_ID;
        }
        else
        {
            return false;
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

	return true;
}