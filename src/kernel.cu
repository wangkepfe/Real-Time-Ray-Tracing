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
#include "reconstruction.cuh"
#include <iomanip>

#define USE_MIS 1
#define USE_TEXTURE_0 0
#define USE_TEXTURE_1 0

extern GlobalSettings* g_settings;

__device__ inline Float3 GetLightSource(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, SurfObj skyBuffer)
{
    // check for termination and hit light
    if (rayState.hitLight == false || rayState.isOccluded == true) { return 0; }

    Float3 lightDir = rayState.dir;
    Float3 L0 = Float3(0);

    // Different light source type
    if (rayState.matType == MAT_SKY)
    {
        // env light
        //Float3 envLightColor = EnvLight(lightDir, cbo.sunDir, cbo.clockTime, rayState.isDiffuseRay);
        //Float3 envLightColor = Float3(0.8f);
        if (cbo.sunDir.y > 0.0f && dot(lightDir, cbo.sunDir) > 0.99999f)
        {
            L0 = Float3(1.0f, 1.0f, 0.9f);
        }
        else if (cbo.sunDir.y < 0.0f && dot(lightDir, -cbo.sunDir) > 0.9999f)
        {
            L0 = Float3(0.9f, 0.95f, 1.0f) * 0.1f;
        }
        else
        {
            Float3 envLightColor = EnvLight2(lightDir, cbo.clockTime, rayState.isDiffuseRay, skyBuffer, Float2(rayState.rand.x, rayState.rand.y));
            L0 = envLightColor;
        }
    }
    else if (rayState.matType == EMISSIVE)
    {
        // local light
        SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];
        L0 = mat.albedo;
    }

    return L0;
}

__device__ inline void GlossySurfaceInteraction(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial)
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
        float surfaceRand = rayState.rand.x;
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
    float sampleLightProbablity,
    Float3& beta)
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
    float lightSamplePdf;
    int lightIdx;

    bool isLightSampled = rayState.rand.z < sampleLightProbablity;
    float sampleLightFactor = 1.0f / (sampleLightProbablity / 0.5f);
    float sampleSurfaceFactor = 1.0f / ((1.0f - sampleLightProbablity) / 0.5f);

    if (isLightSampled)
    {
        bool isLightNotOccluded = SampleLight(cbo, rayState, sceneMaterial, lightSampleDir, lightSamplePdf, isDeltaLight, skyCdf, lightIdx);

        if (isLightNotOccluded == false)
        {
            rayState.isOccluded = true;
            return;
        }
    }

    // surface sample
    Float3 surfSampleDir;

    Float3 surfaceBsdfOverPdf;
    Float3 surfaceSampleBsdf;
    float surfaceSamplePdf = 0;

    Float3 lightSampleSurfaceBsdfOverPdf;
    Float3 lightSampleSurfaceBsdf;
    float lightSampleSurfacePdf = 0;

    Float2 surfaceDiffuseRand2 (rayState.rand.x, rayState.rand.y);
    Float2 surfaceDiffuseRand22 (rayState.rand.z, rayState.rand.w);

    if (rayState.matType == LAMBERTIAN_DIFFUSE)
    {
        LambertianSample(surfaceDiffuseRand2, surfSampleDir, normal);

        if (isDeltaLight == false)
        {
            surfaceBsdfOverPdf = LambertianBsdfOverPdf(albedo);
            surfaceSampleBsdf = LambertianBsdf(albedo);
            surfaceSamplePdf = LambertianPdf(surfSampleDir, normal);
        }

        if (isLightSampled)
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
            MacrofacetReflectionSample(surfaceDiffuseRand2, surfaceDiffuseRand22, rayDir, surfSampleDir, normal, surfaceNormal, surfaceBsdfOverPdf, surfaceSampleBsdf, surfaceSamplePdf, F0, albedo, alpha);
        }

        if (isLightSampled)
        {
            MacrofacetReflection(lightSampleSurfaceBsdfOverPdf, lightSampleSurfaceBsdf, lightSampleSurfacePdf, normal, rayDir, lightSampleDir, F0, albedo, alpha);
        }
    }

    Float3 brdfOverPdf;
    float cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh;

    if (isLightSampled)
    {
        // -------------------------------------- MIS balance heuristic ------------------------------------------
        if (isDeltaLight)
        {
            // if a delta light (or say distant/directional light, typically sun light) is sampled,
            // no surface sample is needed since the weight for surface is zero
            CalculateCosines(rayDir, lightSampleDir, normal, cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh);

            float lightPdf = 1.0f;
			brdfOverPdf = lightSampleSurfaceBsdf * cosThetaWi / lightPdf * sampleLightFactor;
            brdfOverPdf = min3f(brdfOverPdf, Float3(10.0f));

            beta = brdfOverPdf;
            rayState.dir = lightSampleDir;

            rayState.lightIdx = lightIdx;
            rayState.isShadowRay = true;
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

            float misWeight = surfaceSampleWeight / (lightSampleWeight + surfaceSampleWeight);

            float misRand = rayState.rand.w;

            if (misRand < misWeight)
            {
                CalculateCosines(rayDir, surfSampleDir, normal, cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh);

                // choose surface scatter sample
                brdfOverPdf = surfaceBsdfOverPdf  * sampleLightFactor;
                brdfOverPdf = min3f(brdfOverPdf, Float3(10.0f));

                beta = brdfOverPdf;
                rayState.dir = surfSampleDir;
            }
            else
            {
                CalculateCosines(rayDir, lightSampleDir, normal, cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh);

                // choose light sample
                brdfOverPdf = lightSampleSurfaceBsdf * cosThetaWi / lightSamplePdf * sampleLightFactor;
                brdfOverPdf = min3f(brdfOverPdf, Float3(10.0f));

                beta = brdfOverPdf;
                rayState.dir = lightSampleDir;

                rayState.lightIdx = lightIdx;
                rayState.isShadowRay = true;
            }
        }
    }
    else
    {
        CalculateCosines(rayDir, surfSampleDir, normal, cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh);

        // if no light sample, sample surface only
        brdfOverPdf = surfaceBsdfOverPdf * sampleSurfaceFactor;
        brdfOverPdf = min3f(brdfOverPdf, Float3(10.0f));

        beta = brdfOverPdf;
        rayState.dir = surfSampleDir;
    }

    rayState.dir = dot(rayState.normal, rayState.dir) < 0 ? normalize(reflect3f(rayState.dir, rayState.normal)) : rayState.dir;
    rayState.orig = rayState.pos + rayState.offset * rayState.normal;
}

__global__ void PathTrace(ConstBuffer            cbo,
                          SceneGeometry          sceneGeometry,
                          SceneMaterial          sceneMaterial,
                          BlueNoiseRandGenerator randGen,
                          SurfObj                colorBuffer,
                          SurfObj                normalDepthBuffer,
                          SceneTextures          textures,
                          SurfObj                skyBuffer,
                          float*                 skyCdf,
                          SurfObj                motionVectorBuffer,
                          SurfObj                noiseLevelBuffer,
                          SurfObj                indirectLightColorBuffer,
                          SurfObj                indirectLightDirectionBuffer)
{
    // index
    Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
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

    // setup rand gen
    randGen.idx       = idx;
    int sampleNum     = 1;
    int sampleIdx     = 0;
    randGen.sampleIdx = cbo.frameNum * sampleNum + sampleIdx;
    rayState.rand     = randGen.Rand4(0);

    // generate ray
    Float2 sampleUv;
    GenerateRay(rayState.orig, rayState.dir, sampleUv, cbo.camera, idx, Float2(rayState.rand.x, rayState.rand.y), Float2(rayState.rand.z, rayState.rand.w));
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    // save encode normal and depth
    Store2DFloat2(Float2(EncodeNormal_R11_G10_B11(rayState.normal), rayState.depth), normalDepthBuffer, idx);

    // save material
    ushort materialMask = (ushort)rayState.matId;

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

    // glossy only
    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    // glossy + diffuse
    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    DiffuseSurfaceInteraction(cbo, rayState, sceneMaterial, textures, skyCdf, 0.5f, rayState.beta1);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    GlossySurfaceInteraction(cbo, rayState, sceneMaterial);
    DiffuseSurfaceInteraction(cbo, rayState, sceneMaterial, textures, skyCdf, 0.5f, rayState.beta0);
    RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

    // Light
    Float3 L0 = GetLightSource(cbo, rayState, sceneMaterial, skyBuffer);
    Float3 L1 = L0 * rayState.beta0;
    Float3 L2 = L1 * rayState.beta1;

    // write to buffer
    Store2DHalf3Ushort1( { L2 , materialMask } , colorBuffer, idx);
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
    if (g_settings->useDynamicResolution)
    {
        const int minRenderWidth = g_settings->minWidth;
        const int minRenderHeight = g_settings->minHeight;

        historyRenderWidth = renderWidth;
        historyRenderHeight = renderHeight;

        if (deltaTime > 1000.0f / (g_settings->targetFps - 1.0f) && renderWidth > minRenderWidth && renderHeight > minRenderHeight)
        {
            renderWidth -= 16;
            renderHeight -= 9;
            cbo.camera.resolution = Float2(renderWidth, renderHeight);
            cbo.camera.update();
        }
        else if (deltaTime < 1000.0f / (g_settings->targetFps + 1.0f) && renderWidth < maxRenderWidth && renderHeight < maxRenderHeight)
        {
            renderWidth += 16;
            renderHeight += 9;
            cbo.camera.resolution = Float2(renderWidth, renderHeight);
            cbo.camera.update();
        }

        static float timerCounter = 0.0f;

        timerCounter += deltaTime;
        if (timerCounter > 1000.0f)
        {
            timerCounter -= 1000.0f;

            std::cout << "current FPS = " << 1000.0f / deltaTime
                      << ", resolution = (" << renderWidth << ", " << renderHeight
                      << "), percentage to window size = " << std::fixed << std::setprecision(2) << (renderWidth / (float)screenWidth * 100.0f) << "%\n";
        }
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
    gridDim      = dim3(divRoundUp(renderWidth, blockDim.x), divRoundUp(renderHeight, blockDim.y), 1);
    bufferSize4  = UInt2(divRoundUp(renderWidth, 4u), divRoundUp(renderHeight, 4u));
	gridDim4     = dim3(divRoundUp(bufferSize4.x, blockDim.x), divRoundUp(bufferSize4.y, blockDim.y), 1);
    bufferSize16 = UInt2(divRoundUp(bufferSize4.x, 4u), divRoundUp(bufferSize4.y, 4u));
	gridDim16    = dim3(divRoundUp(bufferSize16.x, blockDim.x), divRoundUp(bufferSize16.y, blockDim.y), 1);
    bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u), divRoundUp(bufferSize16.y, 4u));
	gridDim64    = dim3(divRoundUp(bufferSize64.x, blockDim.x), divRoundUp(bufferSize64.y, blockDim.y), 1);

    // ------------------------------- Init -------------------------------
    GpuErrorCheck(cudaMemset(d_histogram, 0, 64 * sizeof(uint)));
    GpuErrorCheck(cudaMemset(morton, UINT_MAX, triCountPadded * sizeof(uint)));
    GpuErrorCheck(cudaMemset(tlasMorton, UINT_MAX, BatchSize * sizeof(uint)));

    // ------------------------------- Sky -------------------------------
    Sky<<<dim3(8, 2, 1), dim3(8, 8, 1)>>>(GetBuffer2D(SkyBuffer), skyCdf, Int2(64, 16), sunDir);
    PrefixScan <1024> <<<1, dim3(512, 1, 1), 1024 * sizeof(float)>>> (skyCdf);

    // ----------------------------------------------- Build bottom level BVH -------------------------------------------------
    // ------------------------------- Update Geometry -----------------------------------
    // out: triangles, aabbs, morton codes
    UpdateSceneGeometry <KernelSize, KernalBatchSize> <<< batchCount, KernelSize >>>
        (constTriangles, triangles, aabbs, morton, triCountArray, clockTime);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("constTriangles.csv"  , constTriangles  , triCountPadded);
        DebugPrintFile("triangles.csv"       , triangles       , triCountPadded);
        DebugPrintFile("aabbs.csv"           , aabbs           , triCountPadded);
        DebugPrintFile("morton.csv"          , morton          , triCountPadded);
    }
    #endif

    // ------------------------------- Radix Sort -----------------------------------
    // in: morton code; out: reorder idx
    RadixSort <KernelSize, KernalBatchSize> <<< batchCount, KernelSize >>>
        (morton, reorderIdx);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("morton2.csv", morton, triCountPadded);
        DebugPrintFile("reorderIdx.csv", reorderIdx, triCountPadded);
    }
    #endif

    // ------------------------------- Build LBVH -----------------------------------
    // in: aabbs, morton code, reorder idx; out: lbvh
    BuildLBVH <KernelSize, KernalBatchSize> <<< batchCount , KernelSize>>>
        (bvhNodes, aabbs, morton, reorderIdx, triCountArray);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("bvhNodes.csv", bvhNodes, triCountPadded);
    }
    #endif

    // ----------------------------------------------- Build top level BVH -------------------------------------------------------------
    UpdateTLAS <KernelSize, KernalBatchSize, BatchSize> <<< 1 , KernelSize>>>
        (bvhNodes, tlasAabbs, tlasMorton, batchCountArray, triCountPadded);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_aabbs.csv"           , tlasAabbs           , BatchSize);
        DebugPrintFile("TLAS_morton.csv"          , tlasMorton          , BatchSize);
    }
    #endif

    RadixSort <KernelSize, KernalBatchSize> <<< 1, KernelSize >>>
        (tlasMorton, tlasReorderIdx);

    #if DEBUG_FRAME > 0
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_reorderIdx.csv"       , tlasReorderIdx      , BatchSize);
        DebugPrintFile("TLAS_morton2.csv"          , tlasMorton          , BatchSize);
    }
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    #endif

    BuildLBVH <KernelSize, KernalBatchSize> <<< 1 , KernelSize>>>
        (tlasBvhNodes, tlasAabbs, tlasMorton, tlasReorderIdx, batchCountArray);

    #if DEBUG_FRAME > 0
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_bvhNodes.csv"       , tlasBvhNodes      , BatchSize);
    }
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    #endif

    auto colorBufferA                 = GetBuffer2D(RenderColorBuffer);
    auto colorBufferB                 = GetBuffer2D(AccumulationColorBuffer);
    auto colorBufferC                 = GetBuffer2D(ScaledColorBuffer);
    auto motionVectorBuffer           = GetBuffer2D(MotionVectorBuffer);
    auto noiseLevelBuffer             = GetBuffer2D(NoiseLevelBuffer);
    auto normalDepthBufferA           = GetBuffer2D(RenderNormalDepthBuffer);
    auto normalDepthBufferB           = GetBuffer2D(HistoryNormalDepthBuffer);
    auto colorBuffer4                 = GetBuffer2D(ColorBuffer4);
    auto colorBuffer16                = GetBuffer2D(ColorBuffer16);
    auto colorBuffer64                = GetBuffer2D(ColorBuffer64);
    auto indirectLightColorBuffer     = GetBuffer2D(IndirectLightColorBuffer);
    auto indirectLightDirectionBuffer = GetBuffer2D(IndirectLightDirectionBuffer);

    // ------------------------------- Path Tracing -------------------------------
    PathTrace<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 8, 1)>>>(
        cbo,
        d_sceneGeometry,
        d_sceneMaterial,
        d_randGen,
        colorBufferA,
        normalDepthBufferA,
        sceneTextures,
        GetBuffer2D(SkyBuffer),
        skyCdf,
        motionVectorBuffer,
        noiseLevelBuffer,
        indirectLightColorBuffer,
        indirectLightDirectionBuffer);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    #endif

    // ------------------------------- Denoising -------------------------------
    // update history camera
    cbo.historyCamera.Setup(cbo.camera);

    // SpatialFilterReconstruction5x5<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(colorBufferA, sceneMaterial, bufferDim);

    // CalculateTileNoiseLevel<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 4, 1)>>>(colorBufferA, normalDepthBufferA, noiseLevelBuffer, bufferDim);

    // if (cbo.frameNum != 1)
    // {
    //     TemporalFilter <<<gridDim, blockDim>>>
    //         (colorBufferA, colorBufferB, normalDepthBufferA, normalDepthBufferB, motionVectorBuffer, noiseLevelBuffer, bufferDim, historyDim);
    // }

    // SpatialFilter5x5<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>
    //     (colorBufferA, normalDepthBufferA, noiseLevelBuffer, bufferDim);

    // CopyToHistoryBuffer<<<gridDim, blockDim>>>(colorBufferA, normalDepthBufferA, colorBufferB, normalDepthBufferB, bufferDim);

    //CalculateTileNoiseLevel<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 4, 1)>>>(colorBufferA, normalDepthBufferA, noiseLevelBuffer, bufferDim);

    //SpatialFilter10x10<<<gridDim, blockDim>>>(colorBufferA, normalDepthBufferA, noiseLevelBuffer, bufferDim);

    //TileNoiseLevelVisualize<<<gridDim, blockDim>>>(colorBufferA, noiseLevelBuffer, bufferDim);

    // ------------------------------- post processing -------------------------------
    // Histogram
    DownScale4 <<<gridDim, blockDim>>>(colorBufferA, colorBuffer4, bufferDim);
    DownScale4 <<<gridDim4, blockDim>>>(colorBuffer4, colorBuffer16, bufferSize4);
    DownScale4 <<<gridDim16, blockDim>>>(colorBuffer16, colorBuffer64, bufferSize16);

    Histogram2<<<1, dim3(min(bufferSize64.x, 32), min(bufferSize64.y, 32), 1)>>>(/*out*/d_histogram, /*in*/colorBuffer64 , bufferSize64);

    // Exposure
    AutoExposure<<<1, 1>>>(/*out*/d_exposure, /*in*/d_histogram, (float)(bufferSize64.x * bufferSize64.y), deltaTime);

    // Bloom
    // BloomGuassian<<<dim3(divRoundUp(bufferSize4.x, 12), divRoundUp(bufferSize4.y, 12), 1), dim3(16, 16, 1)>>>(bloomBuffer4, colorBuffer4, bufferSize4, d_exposure);
    // BloomGuassian<<<dim3(divRoundUp(bufferSize16.x, 12), divRoundUp(bufferSize16.y, 12), 1), dim3(16, 16, 1)>>>(bloomBuffer16, colorBuffer16, bufferSize16, d_exposure);
    // Bloom<<<gridDim, blockDim>>>(colorBufferA, bloomBuffer4, bloomBuffer16, bufferDim, bufferSize4, bufferSize16);

    // Lens flare
    if (sunPos.x > 0 && sunPos.x < 1 && sunPos.y > 0 && sunPos.y < 1 && sunDir.y > -0.0 && dot(sunDir, cbo.camera.dir) > 0)
    {
        sunPos -= Float2(0.5);
        sunPos.x *= (float)renderWidth / (float)renderHeight;
        LensFlarePred<<<1,1>>>(normalDepthBufferA, sunPos, sunUv, colorBufferA, bufferDim, gridDim, blockDim, bufferDim);
    }

    // Tone mapping
    ToneMapping<<<gridDim, blockDim>>>(colorBufferA , bufferDim , d_exposure);

    // Scale to final output
    BicubicScale<<<scaleGridDim, scaleBlockDim>>>(colorBufferC, colorBufferA, outputDim, bufferDim);

    // Sharpening
    SharpeningFilter<<<scaleGridDim, scaleBlockDim>>>(colorBufferC , outputDim);

    // Output
    CopyToOutput<<<scaleGridDim, scaleBlockDim>>>(renderTarget, colorBufferC, outputDim);

    #if DEBUG_FRAME > 0
    if (cbo.frameNum == DEBUG_FRAME)
    {
        // debug
	    CopyFrameBuffer <<<scaleGridDim, scaleBlockDim>>>(dumpFrameBuffer, renderTarget, outputDim);

        writeToPPM("image.ppm", outputDim.x, outputDim.y, dumpFrameBuffer);
    }
    #endif
}