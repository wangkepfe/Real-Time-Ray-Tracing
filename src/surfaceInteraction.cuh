#pragma once

#include "kernel.cuh"
#include "debugUtil.h"
#include "bsdf.cuh"

#define USE_MIS 1
#define USE_TEXTURE_0 0
#define USE_TEXTURE_1 0

__device__ inline void GlossySurfaceInteraction(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, float randNum)
{
    // check for termination and hit light
    if (rayState.hitLight == true || rayState.isDiffuse == true || rayState.isOccluded == true) { return; }

    rayState.isHitProcessed = true;

    if (rayState.matType == PERFECT_REFLECTION)
    {
        // mirror
        rayState.dir = normalize(rayState.dir - rayState.normal * dot(rayState.dir, rayState.normal) * 2.0);
        rayState.orig = rayState.pos + rayState.offset * rayState.normal;
    }
    else if (rayState.matType == PERFECT_FRESNEL_REFLECTION_REFRACTION)
    {
        // glass
        Float3 nextRayDir;
        float rayOffset = rayState.offset;
        float surfaceRand = randNum;
        PerfectReflectionRefraction(1.0, 1.33, rayState.isRayIntoSurface, rayState.normal, rayState.normalDotRayDir, surfaceRand, rayState.dir, nextRayDir, rayOffset);
        rayState.dir = nextRayDir;
        rayState.orig = rayState.pos + rayOffset * rayState.normal;
    }
}

__device__ inline void DiffuseSurfaceInteraction(
    ConstBuffer& cbo,
    RayState& rayState,
    SceneMaterial sceneMaterial,
    SceneTextures textures,
    float* skyCdf,
    float* sunCdf,
    float sampleLightProbablity,
    Float3& beta,
	Float4& randNum,
    Float4& randNum2)
{
    // check for termination and hit light
    if (rayState.hitLight == true || rayState.isDiffuse == false || rayState.isOccluded == true) { return; }

    rayState.isDiffuseRay = true;
    rayState.lightIdx = DEFAULT_LIGHT_ID;
    rayState.isHitProcessed = true;

    // get mat
    SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];

    // texture
    Float3 albedo;
#if USE_TEXTURE_0
    float uvScale = 0.1f;
    if (mat.useTex0)
    {
        if (rayState.matId == 6)
        {
            float4 texColor = tex2D<float4>(textures.array[mat.texId0], - rayState.pos.x * uvScale, - rayState.pos.z * uvScale);
            albedo = Float3(texColor.x, texColor.y, texColor.z);
        }
        else
        {
            float4 texColor = tex2D<float4>(textures.array[mat.texId0], rayState.uv.x * uvScale, rayState.uv.y * uvScale);
            albedo = Float3(texColor.x, texColor.y, texColor.z);
        }
    }
    else
    {
        albedo = mat.albedo;
    }
#else
    albedo = mat.albedo;
#endif

#if USE_INTERPOLATED_FAKE_NORMAL
    Float3 normal = rayState.fakeNormal;
    Float3 surfaceNormal = rayState.normal;
#else
    Float3 normal = rayState.normal;
    Float3 surfaceNormal = rayState.normal;
#endif

#if USE_TEXTURE_1
    if (mat.useTex1)
    {
        float4 texColor = tex2D<float4>(textures.array[mat.texId1], rayState.uv.x * uvScale, rayState.uv.y * uvScale);
        Float3 texNormal = Float3(texColor.x - 0.5, texColor.y - 0.5, texColor.z * 0.5);

        Float3 tangent = Float3(0, 1, 0);

        if (normal.y > 1.0f - 1e-3f)
            tangent = Float3(1, 0, 0);

        Float3 bitangent = cross(normal, tangent);
        tangent = cross(normal, bitangent);

        texNormal = normalize(tangent * texNormal.x + bitangent * texNormal.y + normal * texNormal.z);

        normal = texNormal;
        //normal = mixf(normal, texNormal, 0.0f);
    }
#endif

    Float3 rayDir = rayState.dir;

     // light sample
    float isDeltaLight = false;
    Float3 lightSampleDir;
    float lightSamplePdf = 1;
    int lightIdx;

    SampleLight(cbo, rayState, sceneMaterial, lightSampleDir, lightSamplePdf, isDeltaLight, skyCdf, sunCdf, lightIdx, randNum2);

    // surface sample
    Float3 surfSampleDir;

    Float3 surfaceBsdfOverPdf;
    Float3 surfaceSampleBsdf;
    float surfaceSamplePdf = 0;

    Float3 lightSampleSurfaceBsdfOverPdf;
    Float3 lightSampleSurfaceBsdf;
    float lightSampleSurfacePdf = 0;

    Float2 surfaceDiffuseRand2 (randNum.x, randNum.y);
    Float2 surfaceDiffuseRand22 (randNum.z, randNum.w);

    if (rayState.matType == LAMBERTIAN_DIFFUSE)
    {
        LambertianSample(surfaceDiffuseRand2, surfSampleDir, normal);

        {
            surfaceBsdfOverPdf = LambertianBsdfOverPdf(albedo);
            surfaceSampleBsdf = LambertianBsdf(albedo);
            surfaceSamplePdf = LambertianPdf(surfSampleDir, normal);
        }

        {
            lightSampleSurfaceBsdfOverPdf = LambertianBsdfOverPdf(albedo);
            lightSampleSurfaceBsdf = LambertianBsdf(albedo);
            lightSampleSurfacePdf = LambertianPdf(lightSampleDir, normal);
        }

    }
    else if (rayState.matType == MICROFACET_REFLECTION)
    {
        Float3 F0 = mat.F0;
        float alpha = mat.alpha;

        {
            MacrofacetReflectionSample(surfaceDiffuseRand2, surfaceDiffuseRand22, rayDir, surfSampleDir, normal, surfaceNormal, surfaceBsdfOverPdf, surfaceSampleBsdf, surfaceSamplePdf, F0, albedo, alpha);
        }

        {
            MacrofacetReflection(lightSampleSurfaceBsdfOverPdf, lightSampleSurfaceBsdf, lightSampleSurfacePdf, normal, -rayDir, lightSampleDir, F0, albedo, alpha);
        }
    }

    float cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh;

    NAN_DETECTER(lightSampleSurfaceBsdf);
    NAN_DETECTER(surfaceSampleBsdf);
    NAN_DETECTER(surfaceSamplePdf);
    NAN_DETECTER(lightSamplePdf);

    float powerHeuristicSurface = (surfaceSamplePdf * surfaceSamplePdf) / (surfaceSamplePdf * surfaceSamplePdf + lightSamplePdf * lightSamplePdf);

    if (randNum.z < 0.5)
    {
        if (dot(rayState.normal, surfSampleDir) < 0)
        {
            rayState.isOccluded = true;
            return;
        }

        // choose surface scatter sample
        GetCosThetaWi(surfSampleDir, normal, cosThetaWi);

        float finalPdf = surfaceSamplePdf;
        beta = surfaceSampleBsdf * cosThetaWi / max(finalPdf, 1e-10f);

        rayState.dir = surfSampleDir;
    }
    else
    {
        if (randNum.w < powerHeuristicSurface)
        {
            if (dot(rayState.normal, surfSampleDir) < 0)
            {
                rayState.isOccluded = true;
                return;
            }

            // choose surface scatter sample
            GetCosThetaWi(surfSampleDir, normal, cosThetaWi);

            float finalPdf = surfaceSamplePdf;
            beta = surfaceSampleBsdf * cosThetaWi / max(finalPdf, 1e-10f);

            rayState.dir = surfSampleDir;

            // DEBUG_PRINT(finalPdf);
            // DEBUG_PRINT(cosThetaWi);
            // DEBUG_PRINT(surfaceSampleBsdf);
            // DEBUG_PRINT(beta);
        }
        else
        {
            // choose light sample
            if (dot(rayState.normal, lightSampleDir) < 0)
            {
                rayState.isOccluded = true;
                return;
            }

            GetCosThetaWi(lightSampleDir, normal, cosThetaWi);

            float finalPdf = lightSamplePdf;

            beta = lightSampleSurfaceBsdf * cosThetaWi / max(finalPdf, 1e-10f);

            rayState.dir = lightSampleDir;

            rayState.lightIdx = lightIdx;
            rayState.isShadowRay = true;

            // DEBUG_PRINT(finalPdf);
            // DEBUG_PRINT(cosThetaWi);
            // DEBUG_PRINT(lightSampleSurfaceBsdf);
            // DEBUG_PRINT(beta);
        }
    }

    NAN_DETECTER(beta);

    rayState.dir = rayState.dir;
    rayState.orig = rayState.pos + rayState.offset * rayState.normal;
}
