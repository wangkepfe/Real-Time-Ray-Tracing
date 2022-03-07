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
    //SceneTextures textures,
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

#if USE_INTERPOLATED_FAKE_NORMAL
    Float3 normal = rayState.fakeNormal;
    Float3 surfaceNormal = rayState.normal;
#else
    Float3 normal = rayState.normal;
    Float3 surfaceNormal = rayState.normal;
#endif

    // texture
    Float3 albedo;
#if USE_TEXTURE_0
    if (mat.useTex0)
    {
        const float uvScale = 0.5f;
        Float3 albedoX;
        Float3 albedoY;
        Float3 albedoZ;
        {
            Float2 uv(rayState.pos.y, rayState.pos.z);
            uv *= uvScale;
            float4 texColor = tex2D<float4>(textures.array[mat.texId0], uv[0], uv[1]);
            albedoX = Float3(texColor.x, texColor.y, texColor.z);
            albedoX = pow3f(albedoX, 2.2f);
        }
        {
            Float2 uv(rayState.pos.x, rayState.pos.z);
            uv *= uvScale;
            float4 texColor = tex2D<float4>(textures.array[mat.texId0], uv[0], uv[1]);
            albedoY = Float3(texColor.x, texColor.y, texColor.z);
            albedoY = pow3f(albedoY, 2.2f);
        }
        {
            Float2 uv(rayState.pos.x, rayState.pos.y);
            uv *= uvScale;
            float4 texColor = tex2D<float4>(textures.array[mat.texId0], uv[0], uv[1]);
            albedoZ = Float3(texColor.x, texColor.y, texColor.z);
            albedoZ = pow3f(albedoZ, 2.2f);
        }

        float wx = surfaceNormal.x * surfaceNormal.x;
        float wy = surfaceNormal.y * surfaceNormal.y;
        float wz = surfaceNormal.z * surfaceNormal.z;

        albedo = albedoX * wx + albedoY * wy + albedoZ * wz;
    }
    else
    {
        albedo = mat.albedo;
    }
#else
    albedo = mat.albedo;
#endif

    // float ao = 1.0f;
    // if (mat.useTex2)
    // {
    //     const float uvScale = 0.5f;
    //     float aoX;
    //     float aoY;
    //     float aoZ;
    //     {
    //         Float2 uv(rayState.pos.y, rayState.pos.z);
    //         uv *= uvScale;
    //         aoX = tex2D<float>(textures.array[mat.texId2], uv[0], uv[1]);
    //     }
    //     {
    //         Float2 uv(rayState.pos.x, rayState.pos.z);
    //         uv *= uvScale;
    //         aoY = tex2D<float>(textures.array[mat.texId2], uv[0], uv[1]);
    //     }
    //     {
    //         Float2 uv(rayState.pos.x, rayState.pos.y);
    //         uv *= uvScale;
    //         aoZ = tex2D<float>(textures.array[mat.texId2], uv[0], uv[1]);
    //     }
    //     float wx = surfaceNormal.x * surfaceNormal.x;
    //     float wy = surfaceNormal.y * surfaceNormal.y;
    //     float wz = surfaceNormal.z * surfaceNormal.z;

    //     ao = aoX * wx + aoY * wy + aoZ * wz;
    // }

#if USE_TEXTURE_1
    if (mat.useTex1)
    {
        const float uvScale = 0.5f;
        Float3 normalX;
        Float3 normalY;
        Float3 normalZ;
        {
            Float2 uv(rayState.pos.y, rayState.pos.z);
            uv *= uvScale;
            float4 texColor = tex2D<float4>(textures.array[mat.texId1], uv[0], uv[1]);
            normalX = Float3(texColor.x, texColor.y, texColor.z) - 0.5f;
        }
        {
            Float2 uv(rayState.pos.x, rayState.pos.z);
            uv *= uvScale;
            float4 texColor = tex2D<float4>(textures.array[mat.texId1], uv[0], uv[1]);
            normalY = Float3(texColor.x, texColor.y, texColor.z) - 0.5f;
        }
        {
            Float2 uv(rayState.pos.x, rayState.pos.y);
            uv *= uvScale;
            float4 texColor = tex2D<float4>(textures.array[mat.texId1], uv[0], uv[1]);
            normalZ = Float3(texColor.x, texColor.y, texColor.z) - 0.5f;
        }

        float wx = surfaceNormal.x * surfaceNormal.x;
        float wy = surfaceNormal.y * surfaceNormal.y;
        float wz = surfaceNormal.z * surfaceNormal.z;

        Float3 texNormal = normalize(normalX * wx + normalY * wy + normalZ * wz);

        Float3 w = Float3(1, 0, 0);
        w = normalize(cross(normal, w));
        Float3 u = cross(normal, w);
        Float3 v = cross(normal, u);
        texNormal = normalize(u * texNormal.x + v * texNormal.y + normal * texNormal.z);

        const float normalMapStrength = 1.0f;

        normal = lerp3f(normal, texNormal, normalMapStrength);

        if (dot(normal, surfaceNormal) < 0)
        {
            normal = -normal;
        }

        rayState.fakeNormal = normal;
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

    constexpr float minFinalPdf = 1e-5f;
    constexpr float freeBounceProbability = 0.5f;

    if (randNum.z < freeBounceProbability)
    {
        if (dot(rayState.normal, surfSampleDir) < 0)
        {
            rayState.isOccluded = true;
            return;
        }

        // choose surface scatter sample
        GetCosThetaWi(surfSampleDir, normal, cosThetaWi);

        float finalPdf = surfaceSamplePdf;
        beta = surfaceSampleBsdf * cosThetaWi / max(finalPdf, minFinalPdf);

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
            beta = surfaceSampleBsdf * cosThetaWi / max(finalPdf, minFinalPdf);

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

            beta = lightSampleSurfaceBsdf * cosThetaWi / max(finalPdf, minFinalPdf);

            rayState.dir = lightSampleDir;

            rayState.lightIdx = lightIdx;
            rayState.isShadowRay = true;

            // DEBUG_PRINT(lightSampleDir);
            // DEBUG_PRINT(rayState.normal);
            // DEBUG_PRINT(finalPdf);
            // DEBUG_PRINT(cosThetaWi);
            // DEBUG_PRINT(lightSampleSurfaceBsdf);
            // DEBUG_PRINT(beta);
        }
    }

    rayState.albedo *= albedo;

    NAN_DETECTER(beta);

    rayState.dir = rayState.dir;
    rayState.orig = rayState.pos + rayState.offset * rayState.normal;
}