#pragma once

#include "kernel.cuh"
#include "debugUtil.h"
#include "bsdf.cuh"

#define TOTAL_LIGHT_MAX_COUNT 8

inline __device__ Float3 EnvLight2(const Float3& raydir, float clockTime, bool isDiffuseRay, SurfObj skyBuffer, Float2 blueNoise)
{
	Float2 jitterSize = Float2(1.0f) / Float2(SKY_WIDTH, SKY_HEIGHT);
	Float2 jitter = (blueNoise * jitterSize - jitterSize * 0.5f);

	Float3 color = SampleBicubicSmoothStep(skyBuffer, Load2DFloat4ToFloat3ForSky, EqualAreaMap(raydir), Int2(SKY_WIDTH, SKY_HEIGHT));
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
    const int envLightIdx = 0;
	//const int sunLightIdx = 0;
#endif

    const Float3& normal = rayState.normal;

	float lightChoosePdf;
	int sampledIdx;

	int indexRemap[TOTAL_LIGHT_MAX_COUNT] = {};
	int i = 0;
	int idx = 0;

    indexRemap[idx++] = i++; // env light
    //indexRemap[idx++] = i++; // sun/moon light

	// choose light
    float chooseLightRand = randNum[0];
	int sampledValue = (int)floorf(chooseLightRand * idx);
	sampledIdx = indexRemap[sampledValue];
	lightChoosePdf = 1.0f;// / (float)idx;

	// sample
	Float2 lightSampleRand2(randNum[1], randNum[2]);

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
    if (sampledIdx == envLightIdx)
	{
        float maxSkyCdf = skyCdf[SKY_SIZE - 1];
        float sampledSkyValue = lightSampleRand2[0] * maxSkyCdf;

        // DEBUG_PRINT(maxSkyCdf);
        // DEBUG_PRINT(sampledSkyValue);

        int left = 0;
        int right = SKY_SIZE - 2;
        int mid;
        while (right - left > 1)
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
        mid = left;

        int sampledSkyIdx = mid + 1;
        // DEBUG_PRINT(mid);
        // DEBUG_PRINT(skyCdf[mid + 1]);
        // DEBUG_PRINT(skyCdf[mid]);
        // DEBUG_PRINT(maxSkyCdf);
        float sampledSkyPdf = (skyCdf[mid + 1] - skyCdf[mid]) / maxSkyCdf; // choose 1 from 1024 tiles
        // DEBUG_PRINT(sampledSkyPdf);
        sampledSkyPdf = sampledSkyPdf * SKY_SIZE / TWO_PI; // each tile has area 2Pi / 1024, pdf = 1/area = 1024 / 2Pi

        // DEBUG_PRINT(sampledSkyIdx);
        // DEBUG_PRINT(sampledSkyPdf);

        // index to 2D coordinates
        float u = ((sampledSkyIdx % SKY_WIDTH) + 0.5f) / SKY_WIDTH;
        float v = ((sampledSkyIdx / SKY_WIDTH) + 0.5f) / SKY_HEIGHT;

        // DEBUG_PRINT(u);
        // DEBUG_PRINT(v);

        // hemisphere projection
        float z = v;
        float r = sqrtf(1 - v * v);
        float phi = TWO_PI * u;
        Float3 rayDir(r * cosf(phi), z, r * sinf(phi));

        // DEBUG_PRINT(rayDir);
        lightSampleDir = rayDir;
        lightSamplePdf = sampledSkyPdf * lightChoosePdf;

        lightIdx = ENV_LIGHT_ID;
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

__device__ inline Float3 GetLightSource(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, SurfObj skyBuffer, Float4& randNum)
{
    // check for termination and hit light
    if (rayState.hitLight == false || rayState.isOccluded == true) { return 0; }

    Float3 lightDir = rayState.dir;
    Float3 L0 = Float3(0);

    // Different light source type
    if (rayState.matType == MAT_SKY)
    {
        Float3 envLightColor = EnvLight2(lightDir, cbo.clockTime, rayState.isDiffuseRay, skyBuffer, Float2(randNum.x, randNum.y));
        L0 = envLightColor;
    }
    else if (rayState.matType == EMISSIVE)
    {
        // local light
        SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];
        L0 = mat.albedo;
    }

    return L0;
}
