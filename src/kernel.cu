#include "kernel.cuh"
#include "debug_util.cuh"
#include "geometry.cuh"
#include "bsdf.cuh"
#include "sampler.cuh"
#include "common.cuh"
#include "sky.cuh"
#include "postprocessing.cuh"
#include "raygen.cuh"
#include "traverse.cuh"
#include "light.cuh"
#include "updateGeometry.cuh"
#include "radixSort.cuh"
#include "buildBVH.cuh"
#include "temporalDenoising.cuh"

#define USE_MIS 1
#define USE_TEXTURE_0 1
#define USE_TEXTURE_1 0

const float TargetFPS = 60.0f;

__device__ inline void LightShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, TexObj skyTex)
{
    // check for termination and hit light
    if (rayState.hitLight == false) { return; }

    Float3 beta = rayState.beta;
    Float3 lightDir = rayState.dir;

    // Different light source type
    if (rayState.matType == MAT_SKY)
    {
        // env light
        //Float3 envLightColor = EnvLight(lightDir, cbo.sunDir, cbo.clockTime, rayState.isDiffuseRay);
        //Float3 envLightColor = Float3(0.8f);
        if (cbo.sunDir.y > 0.0f && dot(lightDir, cbo.sunDir) > 0.99999f)
        {
            rayState.L += Float3(1.0f, 1.0f, 0.9f) * beta;
        }
        else if (cbo.sunDir.y < 0.0f && dot(lightDir, -cbo.sunDir) > 0.9999f)
        {
            rayState.L += Float3(0.9f, 0.95f, 1.0f) * 0.1f * beta;
        }
        else
        {
            Float3 envLightColor = EnvLight2(lightDir, cbo.clockTime, rayState.isDiffuseRay, skyTex);
            rayState.L += envLightColor * beta;
        }
    }
    else if (rayState.matType == EMISSIVE)
    {
        // local light
        SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];
        Float3 L0 = mat.albedo;
        rayState.L += L0 * beta;
    }
}

__device__ inline void GlossyShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, BlueNoiseRandGenerator& randGen, int loopIdx)
{
    // check for termination and hit light
    if (rayState.hitLight == true || rayState.isDiffuse == true) { return; }

    if (rayState.matType == PERFECT_REFLECTION)
    {
        // mirror
        rayState.dir = normalize(rayState.dir - rayState.normal * dot(rayState.dir, rayState.normal) * 2.0);
        rayState.orig = rayState.pos + rayState.offset * rayState.normal;

        if (loopIdx == 0) { rayState.bounceLimit = 3; }
    }
    else if (rayState.matType == PERFECT_FRESNEL_REFLECTION_REFRACTION)
    {
        // glass
        Float3 nextRayDir;
        float rayOffset = rayState.offset;
        float surfaceRand = randGen.Rand(4 + loopIdx * 6 + 0);
        PerfectReflectionRefraction(1.0, 1.33, rayState.isRayIntoSurface, rayState.normal, rayState.normalDotRayDir, surfaceRand, rayState.dir, nextRayDir, rayOffset);
        rayState.dir = nextRayDir;
        rayState.orig = rayState.pos + rayOffset * rayState.normal;

        if (loopIdx == 0) { rayState.bounceLimit = 4; }
    }
}

__device__ inline void DiffuseShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, SceneTextures textures, float* skyCdf, BlueNoiseRandGenerator& randGen, int loopIdx)
{
    // check for termination and hit light
    if (rayState.hitLight == true || rayState.isDiffuse == false) { return; }

    rayState.isDiffuseRay = true;
    rayState.lightIdx = DEFAULT_LIGHT_ID;

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
    float lightSamplePdf;
    int lightIdx;

#if USE_MIS
    bool isLightSampled = SampleLight(cbo, rayState, sceneMaterial, lightSampleDir, lightSamplePdf, isDeltaLight, skyCdf, randGen, loopIdx, lightIdx);
#else
    bool isLightSampled = false;
#endif

    // surface sample
    Float3 surfSampleDir;

    Float3 surfaceBsdfOverPdf;
    Float3 surfaceSampleBsdf;
    float surfaceSamplePdf = 0;

    Float3 lightSampleSurfaceBsdfOverPdf;
    Float3 lightSampleSurfaceBsdf;
    float lightSampleSurfacePdf = 0;

    Float2 surfaceDiffuseRand2 = randGen.Rand2(4 + loopIdx * 6 + 0);

    if (rayState.matType == LAMBERTIAN_DIFFUSE)
    {
        LambertianSample(surfaceDiffuseRand2, surfSampleDir, normal);

        if (isDeltaLight == false)
        {
            surfaceBsdfOverPdf = LambertianBsdfOverPdf(albedo);
            surfaceSampleBsdf = LambertianBsdf(albedo);
            surfaceSamplePdf = LambertianPdf(surfSampleDir, normal);
        }

        if (isLightSampled == true)
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

        if (isDeltaLight == false)
        {
            MacrofacetReflectionSample(surfaceDiffuseRand2, rayDir, surfSampleDir, normal, surfaceNormal, surfaceBsdfOverPdf, surfaceSampleBsdf, surfaceSamplePdf, F0, albedo, alpha);
        }

        if (isLightSampled == true)
        {
            MacrofacetReflection(lightSampleSurfaceBsdfOverPdf, lightSampleSurfaceBsdf, lightSampleSurfacePdf, normal, lightSampleDir, rayDir, F0, albedo, alpha);
        }
    }

    // -------------------------------------- MIS balance heuristic ------------------------------------------
    float misRand = randGen.Rand(4 + loopIdx * 6 + 4);

    if (isLightSampled)
    {
        if (isDeltaLight)
        {
            // if a delta light (or say distant/directional light, typically sun light) is sampled,
            // no surface sample is needed since the weight for surface is zero
            rayState.beta *= lightSampleSurfaceBsdf;
            rayState.dir = lightSampleDir;

            rayState.lightIdx = lightIdx;
        }
        else
        {
            // The full equation for MIS is L = sum w_i * f_i / pdf_i
            // which in this case, two samples, one from surface bsdf distribution, one from light distribution
            //
            // L = w_surf * bsdf(dir_surf) / surfaceSamplePdf(dir_surf) + w_light * bsdf(dir_light) / surfaceSamplePdf(dir_light)
            // where w_surf = surfaceSamplePdf(dir_surf) / (surfaceSamplePdf(dir_surf) + lightSamplePdf)
            //       w_light = surfaceSamplePdf(dir_light) / (surfaceSamplePdf(dir_light) + lightSamplePdf)
            //
            // Then it'll become
            // L = bsdf(dir_surf) / (surfaceSamplePdf(dir_surf) + lightSamplePdf) +
            //     bsdf(dir_light) / (surfaceSamplePdf(dir_light) + lightSamplePdf)
            //
            // My algorithm takes bsdf as value and misWeight*pdf as weight,
            // using the weights to choose either sample light or surface.
            // It achieves single ray sample per surface shader with no bias to MIS balance heuristic algorithm

            float surfaceSampleWeight = surfaceSamplePdf / (surfaceSamplePdf + lightSamplePdf * lightSampleSurfacePdf);
            float lightSampleWeight = 1.0f - surfaceSampleWeight;

            //float chooseSurfaceFactor = surfaceSampleWeight / (lightSampleWeight + surfaceSampleWeight);

            //DEBUG_PRINT(lightSamplePdf);
            //DEBUG_PRINT(lightSampleSurfacePdf);
            //DEBUG_PRINT(surfaceSamplePdf);
            //DEBUG_PRINT(chooseSurfaceFactor);

            float misWeight = surfaceSampleWeight / (lightSampleWeight + surfaceSampleWeight);

            if (misRand < misWeight)
            {
                // choose surface scatter sample
                rayState.beta *= min3f(surfaceBsdfOverPdf, Float3(1.0f));
                rayState.dir = surfSampleDir;
            }
            else
            {
                // choose light sample
                rayState.beta *= min3f(lightSampleSurfaceBsdfOverPdf, Float3(1.0f));
                rayState.dir = lightSampleDir;

                rayState.lightIdx = lightIdx;
            }
        }
    }
    else
    {
        // if no light sample condition is met, sample surface only, which is the vanila case
        rayState.beta *= min3f(surfaceBsdfOverPdf, Float3(1.0f));
        rayState.dir = surfSampleDir;
    }

    rayState.dir = dot(rayState.normal, rayState.dir) < 0 ? normalize(reflect3f(rayState.dir, rayState.normal)) : rayState.dir;
    rayState.orig = rayState.pos + rayState.offset * rayState.normal;

    // DEBUG_PRINT(rayState.pos);
    // DEBUG_PRINT(rayState.offset);
    // DEBUG_PRINT(normal);
    // DEBUG_PRINT(rayState.orig);
    // DEBUG_PRINT_BAR
}

__global__ void PathTrace(ConstBuffer            cbo,
                          SceneGeometry          sceneGeometry,
                          SceneMaterial          sceneMaterial,
                          BlueNoiseRandGenerator randGen,
                          SurfObj                colorBuffer,
                          SurfObj                normalDepthBuffer,
                          SceneTextures          textures,
                          TexObj                 skyTex,
                          float*                 skyCdf,
                          SurfObj                motionVectorBuffer,
                          SurfObj                sampleCountBuffer,
                          SurfObj                noiseLevelBuffer)
{
    // index
    Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int i = gridDim.x * blockDim.x * idx.y + idx.x;

    float historyNoiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
    int sampleNum = 1;

    ushort materialMask;
    Float3 outColor = 0;

    #pragma unroll
    for (int sampleIdx = 0; sampleIdx < sampleNum; ++sampleIdx)
    {
        // init ray state
        RayState rayState;
        rayState.i            = i;
        rayState.L            = 0.0;
        rayState.beta         = 1.0;
        rayState.idx          = idx;
        rayState.isDiffuseRay = false;
        rayState.hitLight     = false;
        rayState.bounceLimit  = 2;
        rayState.lightIdx     = DEFAULT_LIGHT_ID;

        // setup rand gen
        randGen.idx = idx;
        randGen.sampleIdx = cbo.frameNum * sampleNum + sampleIdx;

        // generate ray
        Float2 sampleUv;
        GenerateRay(rayState.orig, rayState.dir, sampleUv, cbo.camera, idx, randGen.Rand2(0), randGen.Rand2(2));
        RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

        // first sample first hit
        if (sampleIdx == 0)
        {
            // save encode normal and depth
            Store2DFloat2(Float2(EncodeNormal_R11_G10_B11(rayState.normal), rayState.depth), normalDepthBuffer, idx);

            // save material
            materialMask = (ushort)rayState.matId;

            // calculate motion vector
            Float2 motionVector;
            if (rayState.hit)
            {
                Float2 lastFrameSampleUv = cbo.historyCamera.WorldToScreenSpace(rayState.pos, cbo.camera.tanHalfFov);
                motionVector = lastFrameSampleUv - sampleUv;
            }

            // write motion vector
            motionVector += Float2(0.5f);
            Store2DHalf2(motionVector, motionVectorBuffer, idx);
        }

        // main loop
        int loopIdx = 0;
        for (; loopIdx < rayState.bounceLimit; ++loopIdx)
        {
            GlossyShader(cbo, rayState, sceneMaterial, randGen, loopIdx);
            DiffuseShader(cbo, rayState, sceneMaterial, textures, skyCdf, randGen, loopIdx);
            RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

            //DEBUG_PRINT(loopIdx);
            //DEBUG_PRINT_BAR
        }

        GlossyShader(cbo, rayState, sceneMaterial, randGen, loopIdx);
        LightShader(cbo, rayState, sceneMaterial, skyTex);

        outColor += rayState.L;
    }
    outColor /= (float)sampleNum;

    if (isnan(outColor.x) || isnan(outColor.y) || isnan(outColor.z))
    {
        printf("PathTrace: nan found at (%d, %d)\n", idx.x, idx.y);
        outColor = 0;
    }

    // adaptive sampling debug
    // if (sampleNum == 2)
    // {
    //     outColor += Float3(0, 0.3f, 0);
    // }

    // write to buffer
    Store2DHalf3Ushort1( { outColor , materialMask } , colorBuffer, idx);
}

void RayTracer::UpdateFrame()
{
    // timer
    timer.update();
    deltaTime = timer.getDeltaTime();
    clockTime = timer.getTime();
    cbo.clockTime     = clockTime;
    cbo.camera.resolution = Float2(renderWidth, renderHeight);
    cbo.camera.update();

    // dynamic resolution
    if (UseDynamicResolution)
    {
        const int minRenderWidth = 640;
        const int minRenderHeight = 480;

        historyRenderWidth = renderWidth;
        historyRenderHeight = renderHeight;

        if (deltaTime > 1000.0f / (TargetFPS - 1.0f) && renderWidth > minRenderWidth && renderHeight > minRenderHeight)
        {
            renderWidth -= 16;
            renderHeight -= 9;
            cbo.camera.resolution = Float2(renderWidth, renderHeight);
            cbo.camera.update();
        }
        else if (deltaTime < 1000.0f / (TargetFPS + 1.0f) && renderWidth < maxRenderWidth && renderHeight < maxRenderHeight)
        {
            renderWidth += 16;
            renderHeight += 9;
            cbo.camera.resolution = Float2(renderWidth, renderHeight);
            cbo.camera.update();
        }
        //std::cout << "current FPS = " << 1000.0f / deltaTime << ", resolution = (" << renderWidth << ", " << renderHeight << ")\n";
    }

    // update camera
    InputControlUpdate();

    // sun dir
    const Float3 axis = normalize(Float3(0.0f, 0.0f, 1.0f));
    const float angle = fmodf(clockTime * TWO_PI / 100.0f, TWO_PI);
    sunDir            = rotate3f(axis, angle, Float3(0.0f, 1.0f, 0.0f).normalized()).normalized();
    cbo.sunDir        = sunDir;

    // frame number
    static int framen = 1;
    cbo.frameNum      = framen++;

    // prepare for lens flare
    sunPos = cbo.camera.WorldToScreenSpace(cbo.camera.pos + sunDir);
    sunUv = floor2(sunPos * Float2(renderWidth, renderHeight));

    // init history camera
    if (cbo.frameNum == 1)
    {
        cbo.historyCamera.Setup(cbo.camera);
    }
}

void RayTracer::draw(SurfObj* renderTarget)
{
    // update frame
    UpdateFrame();

    // dimensions
    Int2 bufferDim(renderWidth, renderHeight);
    Int2 historyDim(historyRenderWidth, historyRenderHeight);
    Int2 outputDim(screenWidth, screenHeight);
    gridDim = dim3(divRoundUp(renderWidth, blockDim.x), divRoundUp(renderHeight, blockDim.y), 1);
    bufferSize4 = UInt2(divRoundUp(renderWidth, 4u), divRoundUp(renderHeight, 4u));
	gridDim4 = dim3(divRoundUp(bufferSize4.x, blockDim.x), divRoundUp(bufferSize4.y, blockDim.y), 1);
    bufferSize16 = UInt2(divRoundUp(bufferSize4.x, 4u), divRoundUp(bufferSize4.y, 4u));
	gridDim16 = dim3(divRoundUp(bufferSize16.x, blockDim.x), divRoundUp(bufferSize16.y, blockDim.y), 1);
    bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u), divRoundUp(bufferSize16.y, 4u));
	gridDim64 = dim3(divRoundUp(bufferSize64.x, blockDim.x), divRoundUp(bufferSize64.y, blockDim.y), 1);

    // ------------------------------- Init -------------------------------
    GpuErrorCheck(cudaMemset(d_histogram, 0, 64 * sizeof(uint)));
    GpuErrorCheck(cudaMemset(morton, UINT_MAX, triCount * sizeof(uint)));
    GpuErrorCheck(cudaMemset(isAabbDone, 0, (triCount - 1) * sizeof(uint)));

    // ------------------------------- Sky -------------------------------
    Sky<<<dim3(8, 2, 1), dim3(8, 8, 1)>>>(skyBuffer, skyCdf, Int2(64, 16), sunDir);
    PrefixScan <1024> <<<1, dim3(512, 1, 1), 1024 * sizeof(float)>>> (skyCdf);

    // ------------------------------- Update Geometry -----------------------------------
    // out: triangles, aabbs, morton codes
    UpdateSceneGeometry <1024, 1> <<< 1, 1024 >>> (constTriangles, triangles, aabbs, sceneBoundingBox, morton, triCount, clockTime);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

    if (cbo.frameNum == 10)
    {
        DebugPrintFile("constTriangles.csv", constTriangles, triCount);
        DebugPrintFile("triangles.csv", triangles, triCount);
        DebugPrintFile("aabbs.csv", aabbs, triCount);
        DebugPrintFile("sceneBoundingBox.csv", sceneBoundingBox, 1);
        DebugPrintFile("morton.csv", morton, BVHcapacity);
    }

    // ------------------------------- Radix Sort -----------------------------------
    // in: morton code; out: reorder idx
    RadixSort <1024> <<< 1, 1024 >>> (morton, reorderIdx);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

    if (cbo.frameNum == 10)
    {
        DebugPrintFile("morton2.csv", morton, BVHcapacity);
        DebugPrintFile("reorderIdx.csv", reorderIdx, BVHcapacity);
    }

    // ------------------------------- Build LBVH -----------------------------------
    // in: aabbs, morton code, reorder idx; out: lbvh
    BuildLBVH <1024, 1> <<< 1 , triCount - 1>>> (bvhNodes, aabbs, morton, reorderIdx, triCount);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

    if (cbo.frameNum == 10)
    {
        DebugPrintFile("bvhNodes.csv", bvhNodes, triCount - 1);
    }

    // ------------------------------- Path Tracing -------------------------------
    PathTrace<<<gridDim, blockDim>>>(cbo, d_sceneGeometry, d_sceneMaterial, d_randGen, colorBufferA, normalDepthBufferA, sceneTextures, skyTex, skyCdf, motionVectorBuffer, sampleCountBuffer, noiseLevelBuffer);

    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

    // ------------------------------- Denoising -------------------------------
    // update history camera
    cbo.historyCamera.Setup(cbo.camera);

    // temporal pass
    if (cbo.frameNum != 1)
    {
        TemporalFilter <<<gridDim, blockDim>>>(colorBufferA, colorBufferB, normalDepthBufferA, normalDepthBufferB, motionVectorBuffer, noiseLevelBuffer, bufferDim, historyDim);
    }

    // spatial pass
    AtousFilter2<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(colorBufferA, colorBufferB, normalDepthBufferA, normalDepthBufferB, sampleCountBuffer, noiseLevelBuffer, bufferDim);

    // ------------------------------- post processing -------------------------------
    // Histogram
    DownScale4 <<<gridDim, blockDim>>>(colorBufferA, colorBuffer4, bufferDim);
    DownScale4 <<<gridDim4, blockDim>>>(colorBuffer4, colorBuffer16, bufferSize4);
    DownScale4 <<<gridDim16, blockDim>>>(colorBuffer16, colorBuffer64, bufferSize16);

    Histogram2<<<1, dim3(min(bufferSize64.x, 32), min(bufferSize64.y, 32), 1)>>>(/*out*/d_histogram, /*in*/colorBuffer64 , bufferSize64);

    // Exposure
    AutoExposure<<<1, 1>>>(/*out*/d_exposure, /*in*/d_histogram, (float)(bufferSize64.x * bufferSize64.y), deltaTime);

    // Bloom
    BloomGuassian<<<dim3(divRoundUp(bufferSize4.x, 12), divRoundUp(bufferSize4.y, 12), 1), dim3(16, 16, 1)>>>(bloomBuffer4, colorBuffer4, bufferSize4, d_exposure);
    BloomGuassian<<<dim3(divRoundUp(bufferSize16.x, 12), divRoundUp(bufferSize16.y, 12), 1), dim3(16, 16, 1)>>>(bloomBuffer16, colorBuffer16, bufferSize16, d_exposure);
    Bloom<<<gridDim, blockDim>>>(colorBufferA, bloomBuffer4, bloomBuffer16, bufferDim, bufferSize4, bufferSize16);

    // Lens flare
    if (sunPos.x > 0 && sunPos.x < 1 && sunPos.y > 0 && sunPos.y < 1 && sunDir.y > -0.0 && dot(sunDir, cbo.camera.dir) > 0)
    {
        sunPos -= Float2(0.5);
        sunPos.x *= (float)renderWidth / (float)renderHeight;
        LensFlarePred<<<1,1>>>(normalDepthBufferA, sunPos, sunUv, colorBufferA, bufferDim, gridDim, blockDim, bufferDim);
    }

    // Tone mapping
    ToneMapping<<<gridDim, blockDim>>>(colorBufferA , bufferDim , d_exposure);

    // Sharpening
    SharpeningFilter<<<gridDim, blockDim>>>(colorBufferA , bufferDim);

    // Scale to final output
    BicubicFilterScale<<<scaleGridDim, scaleBlockDim>>>(/*out*/renderTarget, /*in*/colorBufferA, outputDim, bufferDim);

    if (cbo.frameNum == 10)
    {
        // debug
	    CopyFrameBuffer <<<scaleGridDim, scaleBlockDim>>>(dumpFrameBuffer, renderTarget, outputDim);

        writeToPPM("image.ppm", outputDim.x, outputDim.y, dumpFrameBuffer);
    }
}