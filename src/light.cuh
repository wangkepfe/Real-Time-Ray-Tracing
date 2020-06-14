#pragma once

#include "kernel.cuh"
#include "debug_util.cuh"
#include "bsdf.cuh"

#define TOTAL_LIGHT_MAX_COUNT 8

__device__ inline bool SampleLight(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, Float3& lightSampleDir, float& lightSamplePdf, float& isDeltaLight, float* skyCdf)
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
		if (dot(rayState.normal, vec) > 0 && vec.length2() < 1.0)
			indexRemap[idx++] = i; // sphere light
	}

	indexRemap[idx++] = i++; // sun/moon light
    indexRemap[idx++] = i++; // env light

	// choose light
	int sampledValue = rd(&rayState.rdState[2]) * idx;
	sampledIdx = indexRemap[sampledValue];
	lightChoosePdf = 1.0 / idx;

	// sample
	if (sampledIdx == sunLightIdx)
	{
        if (dot(rayState.normal, cbo.sunDir) > 0)
        {
            Float3 moonDir = cbo.sunDir;
            moonDir        = -moonDir;
            lightSampleDir = cbo.sunDir.y > -0.05 ? cbo.sunDir : moonDir;
            lightSamplePdf = 1.0f;
            isDeltaLight   = true;
        }
        else
        {
            return false;
        }
	}
    else if (sampledIdx == envLightIdx)
	{
        float maxSkyCdf = skyCdf[1023];
        float sampledSkyValue = rd(&rayState.rdState[2]) * maxSkyCdf;

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

            //printf("sampledSkyValue = %f, skyCdf[mid] = %f, skyCdf[mid+1] = %f, sampledSkyIdx = %d, sampledSkyPdf = %f, rayDir = (%f, %f, %f)\n", sampledSkyValue, skyCdf[mid], skyCdf[mid + 1], sampledSkyIdx, sampledSkyPdf, rayDir.x, rayDir.y, rayDir.z);
        }
        else
        {
            return false;
        }
	}
	else
	{
		Sphere sphereLight = sphereLights[sampledIdx];

		Float3 lightDir   = sphereLight.center - rayState.pos;
		float dist2       = lightDir.length2();
		float radius2     = sphereLight.radius * sphereLight.radius;
		float cosThetaMax = sqrtf(max1f(dist2 - radius2, 0)) / sqrtf(dist2);

		Float3 u, v;
		LocalizeSample(lightDir, u, v);
		lightSampleDir = UniformSampleCone(rd2(&rayState.rdState[0], &rayState.rdState[1]), cosThetaMax, u, v, lightDir);
		lightSampleDir = normalize(lightSampleDir);
		lightSamplePdf = UniformConePdf(cosThetaMax) * lightChoosePdf;
	}

	return true;
}