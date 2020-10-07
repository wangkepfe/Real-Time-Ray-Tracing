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
    BlueNoiseRandGenerator& randGen,
    int                     loopIdx,
    int&                    lightIdx)
{
	const int numSphereLight = sceneMaterial.numSphereLights;
	Sphere* sphereLights = sceneMaterial.sphereLights;

	float lightChoosePdf;
	int sampledIdx;

	int indexRemap[TOTAL_LIGHT_MAX_COUNT] = {};
	int i = 0;
	int idx = 0;

	const int sunLightIdx = numSphereLight;
    const int envLightIdx = numSphereLight + 1;

	for (; i < numSphereLight; ++i)
	{
		Float3 vec = sphereLights[i].center - rayState.pos;
        float dist2 = vec.length2();
		if (dot(rayState.normal, vec) > 0 && dist2 < 10.0f)
        {
            indexRemap[idx++] = i; // sphere light
        }
	}

	indexRemap[idx++] = i++; // sun/moon light
    indexRemap[idx++] = i++; // env light

	// choose light
    float chooseLightRand = randGen.Rand(4 + loopIdx * 6 + 5);
	int sampledValue = (int)floorf(chooseLightRand * idx);
	sampledIdx = indexRemap[sampledValue];
	lightChoosePdf = 1.0f / (float)idx;

	// sample
	Float2 lightSampleRand2 = randGen.Rand2(4 + loopIdx * 6 + 2);

	if (sampledIdx == sunLightIdx)
	{
        if (dot(rayState.normal, cbo.sunDir) > 0)
        {
            Float3 moonDir = cbo.sunDir;
            moonDir        = -moonDir;

            lightSampleDir = cbo.sunDir.y > -0.05 ? cbo.sunDir : moonDir;
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

        int left = 0;
        int right = 1022;
        int mid;
        for (int j = 0; j < 8; ++j)
        {
            mid = (left + right) / 2;
            float midVal = skyCdf[mid];

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
        float sampledSkyPdf = (skyCdf[mid + 1] - skyCdf[mid]) / maxSkyCdf * 1024.0f / TWO_PI;

        // index to 2D coordinates
        float u = ((sampledSkyIdx % 64) + 0.5f) / 64;
        float v = ((sampledSkyIdx / 64) + 0.5f) / 16;

        // hemisphere projection
        float z = v;
        float r = sqrtf(1 - v * v);
        float phi = TWO_PI * u;
        Float3 rayDir(r * cosf(phi), z, r * sinf(phi));

        if (dot(rayDir, rayState.normal) > 0)
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
	else
	{
		Sphere sphereLight = sphereLights[sampledIdx];

        // light vector / direction
		Float3 lightVec   = sphereLight.center - rayState.pos;
        Float3 lightDir   = normalize(lightVec);

        // cos theta max
        float dist2       = lightVec.length2();
        float radius2     = sphereLight.radius * sphereLight.radius;
        float cosThetaMax = sqrtf(max1f(dist2 - radius2, 0)) / sqrtf(dist2);

        // sample direction based on light direction and cosThetaMax
        Float3 u, v;
        LocalizeSample(lightDir, u, v);
        lightSampleDir = UniformSampleCone(lightSampleRand2, cosThetaMax, u, v, lightDir);
        lightSampleDir = normalize(lightSampleDir);

        // if direction in sight
        if (dot(rayState.normal, lightSampleDir) > 0)
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

	return true;
}