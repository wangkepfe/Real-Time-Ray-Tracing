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
                          SurfObj                sunBuffer,
                          float*                 sunCdf,
                          SurfObj                motionVectorBuffer,
                          SurfObj                noiseLevelBuffer,
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
    Float4 randNum[4];
    randGen.idx       = idx;
    randGen.sampleIdx = cbo.frameNum * 4 + 0;
    randNum[0] = randGen.Rand4(0);
    randGen.sampleIdx = cbo.frameNum * 4 + 1;
    randNum[1] = randGen.Rand4(0);
    randGen.sampleIdx = cbo.frameNum * 4 + 2;
    randNum[2] = randGen.Rand4(0);
    randGen.sampleIdx = cbo.frameNum * 4 + 3;
    randNum[3] = randGen.Rand4(0);

    // generate ray
    Float2 sampleUv;
    GenerateRay(rayState.orig, rayState.dir, sampleUv, cbo.camera, idx, Float2(randNum[0][0], randNum[0][1]), Float2(randNum[0][2], randNum[0][3]));
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    #if USE_INTERPOLATED_FAKE_NORMAL
    Float3 outputNormal = rayState.fakeNormal;
    #else
    Float3 outputNormal = rayState.normal;
    #endif

    float outputDepth = rayState.depth;
	Float2 outputUv = rayState.uv;

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
    GlossySurfaceInteraction(cbo, rayState, sceneMaterial, randNum[0][0]);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    GlossySurfaceInteraction(cbo, rayState, sceneMaterial, randNum[0][1]);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    // glossy + diffuse
    GlossySurfaceInteraction(cbo, rayState, sceneMaterial, randNum[0][2]);
    DiffuseSurfaceInteraction(cbo, rayState, sceneMaterial, textures, skyCdf, sunCdf, 0.1f, rayState.beta1, randNum[0], randNum[1]);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    GlossySurfaceInteraction(cbo, rayState, sceneMaterial, randNum[0][3]);
    DiffuseSurfaceInteraction(cbo, rayState, sceneMaterial, textures, skyCdf, sunCdf, 0.1f, rayState.beta0, randNum[2], randNum[3]);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    // Light
    Float3 L0 = GetLightSource(cbo, rayState, sceneMaterial, skyBuffer, sunBuffer, randNum[0]);
    Float3 L1 = L0 * rayState.beta0;
    Float3 L2 = L1 * rayState.beta1;

    // write to buffer
    NAN_DETECTER(L0);
    NAN_DETECTER(L2);
    NAN_DETECTER(outputNormal);
    NAN_DETECTER(outputDepth);
    NAN_DETECTER(motionVector);

    L2 = clamp3f(L2, Float3(0.0f), Float3(10.0f));

    Store2DHalf3Ushort1( { L2 , materialMask } , colorBuffer, idx);
    Store2DHalf4(Float4(outputNormal, 0), normalBuffer, idx);
    Store2DHalf1(outputDepth, depthBuffer, idx);
    Store2DHalf2(motionVector, motionVectorBuffer, idx);
}
