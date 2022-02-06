#pragma once

#include "kernel.cuh"
#include "debugUtil.h"
#include "sampler.cuh"
#include "raygen.cuh"
#include "traverse.cuh"
#include "light.cuh"
#include "surfaceInteraction.cuh"

__global__ void PathTrace(ConstBuffer            cbo,
                          SceneGeometry          sceneGeometry,
                          SceneMaterial          sceneMaterial,
                          BlueNoiseRandGenerator randGen,
                          SurfObj                colorBuffer,
                          SurfObj                normalBuffer,
                          SurfObj                depthBuffer,
                          SceneTextures          textures,
                          SurfObj                skyBuffer,
                          float*                 skyCdf,
                          SurfObj                motionVectorBuffer,
                          SurfObj                noiseLevelBuffer,
                          SurfObj                indirectLightColorBuffer,
                          SurfObj                indirectLightDirectionBuffer,
                          Int2                   renderSize)
{
    // index
    Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (idx.x >= renderSize.x || idx.y >= renderSize.y) return;
    int i = gridDim.x * blockDim.x * idx.y + idx.x;

    float historyNoiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));

    // init ray state
    RayState rayState;
    rayState.i              = i;
    rayState.beta0          = Float3(1.0f);
    rayState.beta1          = Float3(1.0f);
    rayState.idx            = idx;
    rayState.isDiffuseRay   = false;
    rayState.hitLight       = false;
    rayState.lightIdx       = DEFAULT_LIGHT_ID;
    rayState.isHitProcessed = true;
    rayState.isOccluded     = false;
    rayState.isShadowRay    = false;
    rayState.normal         = Float3(0, -1, 0);

    // setup rand gen
    randGen.idx       = idx;
    int sampleNum     = 1;
    int sampleIdx     = 0;
    randGen.sampleIdx = cbo.frameNum * sampleNum + sampleIdx;
    rayState.rand     = randGen.Rand4(0);

    // generate ray
    Float2 sampleUv;
    GenerateRay(rayState.orig, rayState.dir, sampleUv, cbo.camera, idx, Float2(rayState.rand.x, rayState.rand.y), Float2(rayState.rand.x, rayState.rand.y));
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    #if USE_INTERPOLATED_FAKE_NORMAL
    Float3 outputNormal = rayState.fakeNormal;
    #else
    Float3 outputNormal = rayState.normal;
    #endif

    float outputDepth = rayState.depth;

    // save material
    ushort materialMask = (ushort)rayState.matId;

    // calculate motion vector
    Float2 motionVector;
    if (rayState.hit)
    {
        Float2 lastFrameSampleUv = cbo.historyCamera.WorldToScreenSpace(rayState.pos, cbo.camera.tanHalfFov);
        motionVector = lastFrameSampleUv - sampleUv;
    }
    motionVector += Float2(0.5f);

    // glossy only
    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    Float3 indirectLightDir;

    // glossy + diffuse
    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    DiffuseSurfaceInteraction(cbo, rayState, sceneMaterial, textures, skyCdf, 0.1f, rayState.beta1, &indirectLightDir);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    DiffuseSurfaceInteraction(cbo, rayState, sceneMaterial, textures, skyCdf, 0.1f, rayState.beta0);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    // Light
    Float3 L0 = GetLightSource(cbo, rayState, sceneMaterial, skyBuffer);
    NAN_DETECTER(L0);
    Float3 L1 = L0 * rayState.beta0;
    Float3 L2 = L1 * rayState.beta1;

    // write to buffer
    NAN_DETECTER(L2);
    NAN_DETECTER(outputNormal);
    NAN_DETECTER(outputDepth);
    NAN_DETECTER(motionVector);
    NAN_DETECTER(L1);
    NAN_DETECTER(indirectLightDir);
    Store2DHalf3Ushort1( { L2 , materialMask } , colorBuffer, idx);
    Store2DHalf4(Float4(outputNormal, 0), normalBuffer, idx);
    Store2DHalf1(outputDepth, depthBuffer, idx);
    Store2DHalf2(motionVector, motionVectorBuffer, idx);
    Store2DHalf4(Float4(L1, 0), indirectLightColorBuffer, idx);
    Store2DHalf4(Float4(indirectLightDir, 0), indirectLightDirectionBuffer, idx);
}
