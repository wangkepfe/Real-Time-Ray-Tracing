#include "kernel.cuh"
#include "debugUtil.h"
#include "scan.cuh"
#include "sky.cuh"
#include "postprocessing.cuh"
#include "updateGeometry.cuh"
#include "radixSort.cuh"
#include "buildBVH.cuh"
#include "temporalDenoising.cuh"
#include "reconstruction.cuh"
#include "settingParams.h"
#include "pathtrace.cuh"
#include "surfaceInteraction.cuh"

extern GlobalSettings* g_settings;

void RayTracer::UpdateFrame()
{
    // frame number
    static int framen = 1;
    cbo.frameNum      = framen++;

    // timer
    timer.update();
    deltaTime = timer.getDeltaTime();
    clockTime = timer.getTime();
    cbo.clockTime     = clockTime;
    cbo.camera.resolution = Float2(renderWidth, renderHeight);
    cbo.camera.update();

    // dynamic resolution
    if (g_settings->useDynamicResolution && cbo.frameNum > 1)
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
    const float angle = fmodf(skyParams.timeOfDay * TWO_PI, TWO_PI);
    sunDir            = rotate3f(axis, angle, Float3(0.0f, 1.0f, 0.0f).normalized()).normalized();
    cbo.sunDir        = sunDir;

    // prepare for lens flare
    sunPos = cbo.camera.WorldToScreenSpace(cbo.camera.pos + sunDir);
    sunUv = floor2(sunPos * Float2(renderWidth, renderHeight));


    // init history camera
    if (cbo.frameNum == 1)
    {
        cbo.historyCamera.Setup(cbo.camera);
        CalculateGaussian3x3();
        CalculateGaussian5x5();
        CalculateGaussian7x7();
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
    Sky<<<dim3(SKY_WIDTH / 8, SKY_HEIGHT / 8, 1), dim3(8, 8, 1)>>>(GetBuffer2D(SkyBuffer), skyPdf, Int2(SKY_WIDTH, SKY_HEIGHT), sunDir, skyParams);
    // ScanSingleBlock <1024, 1, 4> <<<1, dim3(128, 1, 1), 1024 * sizeof(float)>>> (skyCdf, skyCdf);
    Scan(skyPdf, skyCdf, skyCdfScanTmp, SKY_SIZE, SKY_SCAN_BLOCK_SIZE, 1);

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
        DebugPrintFile("TLAS_aabbs.csv"           , tlasAabbs           , batchCount);
        DebugPrintFile("TLAS_morton.csv"          , tlasMorton          , batchCount);
    }
    #endif

    RadixSort <KernelSize, KernalBatchSize> <<< 1, KernelSize >>>
        (tlasMorton, tlasReorderIdx);

    #if DEBUG_FRAME > 0
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_reorderIdx.csv"       , tlasReorderIdx      , batchCount);
        DebugPrintFile("TLAS_morton2.csv"          , tlasMorton          , batchCount);
    }
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    #endif

    BuildLBVH <KernelSize, KernalBatchSize> <<< 1 , KernelSize>>>
        (tlasBvhNodes, tlasAabbs, tlasMorton, tlasReorderIdx, batchCountArray);

    #if DEBUG_FRAME > 0
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_bvhNodes.csv"       , tlasBvhNodes      , batchCount);
    }
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    #endif

    auto colorBufferA                 = GetBuffer2D(RenderColorBuffer);
    auto colorBufferB                 = GetBuffer2D(AccumulationColorBuffer);
    auto colorBufferC                 = GetBuffer2D(ScaledColorBuffer);
    auto motionVectorBuffer           = GetBuffer2D(MotionVectorBuffer);
    auto noiseLevelBuffer             = GetBuffer2D(NoiseLevelBuffer);
    auto noiseLevelBuffer16           = GetBuffer2D(NoiseLevelBuffer16x16);
    auto normalBuffer                 = GetBuffer2D(NormalBuffer);
    auto depthBufferA                 = GetBuffer2D(DepthBuffer);
    auto depthBufferB                 = GetBuffer2D(HistoryDepthBuffer);
    auto colorBuffer4                 = GetBuffer2D(ColorBuffer4);
    auto colorBuffer16                = GetBuffer2D(ColorBuffer16);
    auto colorBuffer64                = GetBuffer2D(ColorBuffer64);
    auto bloomBuffer4                 = GetBuffer2D(BloomBuffer4);
    auto bloomBuffer16                = GetBuffer2D(BloomBuffer16);
    auto indirectLightColorBuffer     = GetBuffer2D(IndirectLightColorBuffer);
    auto indirectLightDirectionBuffer = GetBuffer2D(IndirectLightDirectionBuffer);

    // ------------------------------- Path Tracing -------------------------------
    PathTrace<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 8, 1)>>>(
        cbo,
        d_sceneGeometry,
        d_sceneMaterial,
        d_randGen,
        colorBufferA,
        normalBuffer,
        depthBufferA,
        sceneTextures,
        GetBuffer2D(SkyBuffer),
        skyCdf,
        motionVectorBuffer,
        noiseLevelBuffer,
        indirectLightColorBuffer,
        indirectLightDirectionBuffer,
        bufferDim);

    // ------------------------------- Reconstruct -------------------------------
    if (renderPassSettings.enableDenoiseReconstruct)
    {
        SpatialReconstruction5x5<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
            colorBufferA,
            normalBuffer,
            depthBufferA,
            indirectLightColorBuffer,
            indirectLightDirectionBuffer,
            d_sceneMaterial,
            d_randGen,
            cbo,
            bufferDim);
    }

    // update history camera
    cbo.historyCamera.Setup(cbo.camera);

    // ------------------------------- Temporal Spatial Denoising -------------------------------
    UInt2 noiseLevel16x16Dim(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16));
    if (renderPassSettings.enableTemporalDenoising)
    {
        CalculateTileNoiseLevel<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 4, 1)>>>(
            colorBufferA,
            depthBufferA,
            noiseLevelBuffer,
            bufferDim);


        TileNoiseLevel8x8to16x16<<<dim3(divRoundUp(noiseLevel16x16Dim.x, 8), divRoundUp(noiseLevel16x16Dim.y, 8), 1), dim3(8, 8, 1)>>>(
            noiseLevelBuffer,
            noiseLevelBuffer16);

        if (cbo.frameNum != 1)
        {
            TemporalFilter <<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 8, 1)>>>(
                cbo,
                colorBufferA, colorBufferB,
                normalBuffer,
                depthBufferA, depthBufferB,
                motionVectorBuffer,
                noiseLevelBuffer,
                bufferDim, historyDim);
        }
    }

    if (renderPassSettings.enableLocalSpatialFilter)
    {
        SpatialFilter7x7<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
        cbo,
        colorBufferA,
        normalBuffer,
        depthBufferA,
        noiseLevelBuffer16,
        bufferDim);
    }

    if (renderPassSettings.enableTemporalDenoising)
    {
        CopyToHistoryColorBuffer<<<gridDim, blockDim>>>(colorBufferA, colorBufferB, bufferDim);
    }

    CalculateTileNoiseLevel<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 4, 1)>>>(
        colorBufferA,
        depthBufferA,
        noiseLevelBuffer,
        bufferDim);

    TileNoiseLevel8x8to16x16<<<dim3(divRoundUp(noiseLevel16x16Dim.x, 8), divRoundUp(noiseLevel16x16Dim.y, 8), 1), dim3(8, 8, 1)>>>(
        noiseLevelBuffer,
        noiseLevelBuffer16);

    if (renderPassSettings.enableNoiseLevelVisualize)
    {
        TileNoiseLevelVisualize<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
            colorBufferA,
            noiseLevelBuffer16,
            bufferDim);
    }

    if (renderPassSettings.enableWideSpatialFilter)
    {
        SpatialFilterGlobal5x5<3><<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(cbo,colorBufferA,normalBuffer,depthBufferA,noiseLevelBuffer16,bufferDim);
        SpatialFilterGlobal5x5<6><<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(cbo,colorBufferA,normalBuffer,depthBufferA,noiseLevelBuffer16,bufferDim);
        SpatialFilterGlobal5x5<12><<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(cbo,colorBufferA,normalBuffer,depthBufferA,noiseLevelBuffer16,bufferDim);
    }

    if (renderPassSettings.enableTemporalDenoising2)
    {
        if (cbo.frameNum != 1)
        {
            TemporalFilter2 <<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 8, 1)>>>(
                cbo,
                colorBufferA, GetBuffer2D(HistoryColorBuffer),
                normalBuffer,
                depthBufferA, depthBufferB,
                motionVectorBuffer,
                noiseLevelBuffer,
                bufferDim, historyDim);
        }
        CopyToHistoryColorDepthBuffer<<<gridDim, blockDim>>>(colorBufferA, depthBufferA, GetBuffer2D(HistoryColorBuffer), depthBufferB, bufferDim);
    }

    if (0)
    {
        GpuErrorCheck(cudaDeviceSynchronize());
	    GpuErrorCheck(cudaPeekAtLastError());
    }

    // ------------------------------- post processing -------------------------------
    if (renderPassSettings.enablePostProcess)
    {
        // Histogram
        DownScale4 <<<gridDim, blockDim>>>(colorBufferA, colorBuffer4, bufferDim);
        DownScale4 <<<gridDim4, blockDim>>>(colorBuffer4, colorBuffer16, bufferSize4);
        DownScale4 <<<gridDim16, blockDim>>>(colorBuffer16, colorBuffer64, bufferSize16);

        Histogram2<<<1, dim3(min(bufferSize64.x, 32), min(bufferSize64.y, 32), 1)>>>(/*out*/d_histogram, /*in*/colorBuffer64 , bufferSize64);

        // Exposure
        AutoExposure<<<1, 1>>>(/*out*/d_exposure, /*in*/d_histogram, (float)(bufferSize64.x * bufferSize64.y), deltaTime);

        // Bloom
        if (renderPassSettings.enableBloomEffect)
        {
            BloomGuassian<<<dim3(divRoundUp(bufferSize4.x, 12), divRoundUp(bufferSize4.y, 12), 1), dim3(16, 16, 1)>>>(bloomBuffer4, colorBuffer4, bufferSize4, d_exposure);
            BloomGuassian<<<dim3(divRoundUp(bufferSize16.x, 12), divRoundUp(bufferSize16.y, 12), 1), dim3(16, 16, 1)>>>(bloomBuffer16, colorBuffer16, bufferSize16, d_exposure);
            Bloom<<<gridDim, blockDim>>>(colorBufferA, bloomBuffer4, bloomBuffer16, bufferDim, bufferSize4, bufferSize16);
        }

        // Lens flare
        if (renderPassSettings.enableLensFlare)
        {
            if (sunPos.x > 0 && sunPos.x < 1 && sunPos.y > 0 && sunPos.y < 1 && sunDir.y > -0.0 && dot(sunDir, cbo.camera.dir) > 0)
            {
                sunPos -= Float2(0.5);
                sunPos.x *= (float)renderWidth / (float)renderHeight;
                LensFlarePred<<<1,1>>>(depthBufferA, sunPos, sunUv, colorBufferA, bufferDim, gridDim, blockDim, bufferDim);
            }
        }

        // Tone mapping
        if (renderPassSettings.enableToneMapping)
        {
            ToneMapping<<<gridDim, blockDim>>>(colorBufferA , bufferDim , d_exposure);
        }
    }

    // Scale to final output
    BicubicScale<<<scaleGridDim, scaleBlockDim>>>(colorBufferC, colorBufferA, outputDim, bufferDim);

    if (renderPassSettings.enablePostProcess)
    {
        // Sharpening
        if (renderPassSettings.enableSharpening)
        {
            SharpeningFilter<<<scaleGridDim, scaleBlockDim>>>(colorBufferC , outputDim);
        }
    }

    // Output
    CopyToOutput<<<scaleGridDim, scaleBlockDim>>>(renderTarget, colorBufferC, d_randGen, cbo, outputDim);

    // Debug
    #if DUMP_FRAME_NUM > 0
    if (cbo.frameNum == DUMP_FRAME_NUM || cbo.frameNum == DEBUG_FRAME)
    {
        // debug
	    CopyFrameBuffer <<<scaleGridDim, scaleBlockDim>>>(dumpFrameBuffer, renderTarget, outputDim);

        std::string outputName = "outputImage_frame_" + std::to_string(cbo.frameNum) + "_" + Timer::getTimeString() + ".ppm";

        writeToPPM(outputName, outputDim.x, outputDim.y, dumpFrameBuffer);
    }
    #endif
}