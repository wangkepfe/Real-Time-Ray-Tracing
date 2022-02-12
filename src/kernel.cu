#include "kernel.cuh"
#include "debugUtil.h"
#include "scan.cuh"
#include "sky.cuh"
#include "postprocessing.cuh"
#include "settingParams.h"
#include "pathtrace.cuh"

extern GlobalSettings* g_settings;

__global__ void CopyFrameBuffer(
	uchar4*  dumpFrameBuffer,
	SurfObj* renderTarget,
	Int2     outSize)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= outSize.x || idx.y >= outSize.y) return;

	uchar4 val = surf2Dread<uchar4>(renderTarget[0], idx.x * 4 * sizeof(uchar1), idx.y, cudaBoundaryModeClamp);

	dumpFrameBuffer[idx.x + idx.y * outSize.x] = val;
}

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
    const Float3 axis = normalize(Float3(0.0f, cos(skyParams.sunAxisAngle * Pi_over_180), sin(skyParams.sunAxisAngle * Pi_over_180)));
    const float angle = fmodf(skyParams.timeOfDay * M_PI, TWO_PI);
    sunDir            = rotate3f(axis, angle, cross(Float3(0, 1, 0), axis)).normalized();
    cbo.sunDir        = sunDir;

    // prepare for lens flare
    sunPos = cbo.camera.WorldToScreenSpace(cbo.camera.pos + sunDir);
    sunUv = floor2(sunPos * Float2(renderWidth, renderHeight));

    if (cbo.frameNum == 1)
    {
        // init history camera
        cbo.historyCamera.Setup(cbo.camera);
        CalculateGaussian3x3();
        CalculateGaussian5x5();
        CalculateGaussian7x7();

        // init sky
        UploadSkyConstantBuffer();
    }
}

void RayTracer::draw(SurfObj* renderTarget)
{
    // update frame
    UpdateFrame();

    // dimensions
    Int2 bufferDim(renderWidth        , renderHeight);
    Int2 historyDim(historyRenderWidth, historyRenderHeight);
    Int2 outputDim(screenWidth        , screenHeight);

    gridDim      = dim3 (divRoundUp(renderWidth   , blockDim.x ) , divRoundUp(renderHeight           , blockDim.y ), 1 );
    bufferSize4  = UInt2(divRoundUp(renderWidth   , 4u         ) , divRoundUp(renderHeight           , 4u         )    );
	gridDim4     = dim3 (divRoundUp(bufferSize4.x , blockDim.x ) , divRoundUp (bufferSize4.y         , blockDim.y ), 1 );
    bufferSize16 = UInt2(divRoundUp(bufferSize4.x , 4u         ) , divRoundUp         (bufferSize4.y , 4u         )    );
	gridDim16    = dim3 (divRoundUp(bufferSize16.x, blockDim.x ) , divRoundUp(bufferSize16.y         , blockDim.y ), 1 );
    bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u         ) , divRoundUp        (bufferSize16.y , 4u         )    );
	gridDim64    = dim3 (divRoundUp(bufferSize64.x, blockDim.x ) , divRoundUp(bufferSize64.y         , blockDim.y ), 1 );

    // ------------------------------- Init -------------------------------
    GpuErrorCheck(cudaMemset(d_histogram, 0, 64 * sizeof(uint)));
    GpuErrorCheck(cudaMemset(morton, UINT_MAX, triCountPadded * sizeof(uint)));
    GpuErrorCheck(cudaMemset(tlasMorton, UINT_MAX, BatchSize * sizeof(uint)));

    // ------------------------------- Sky -------------------------------
    if (skyParams.needRegenerate)
    {
        Sky<<<dim3(SKY_WIDTH / 8, SKY_HEIGHT / 8, 1), dim3(8, 8, 1)>>>(GetBuffer2D(SkyBuffer), skyPdf, Int2(SKY_WIDTH, SKY_HEIGHT), sunDir, skyParams);
        Scan(skyPdf, skyCdf, skyCdfScanTmp, SKY_SIZE, SKY_SCAN_BLOCK_SIZE, 1);

        skyParams.needRegenerate = false;
    }

    // ------------------------------- BVH -------------------------------
    BuildBvhLevel1();
    BuildBvhLevel2();

    // ------------------------------- Path Tracing -------------------------------
    PathTrace<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 8, 1)>>>(
        cbo,
        d_sceneGeometry,
        d_sceneMaterial,
        d_randGen,
        GetBuffer2D(RenderColorBuffer),
        GetBuffer2D(NormalBuffer),
        GetBuffer2D(DepthBuffer),
        sceneTextures,
        GetBuffer2D(SkyBuffer),
        skyCdf,
        GetBuffer2D(MotionVectorBuffer),
        GetBuffer2D(NoiseLevelBuffer),
        bufferDim);

    // update history camera
    cbo.historyCamera.Setup(cbo.camera);

    // ------------------------------- Temporal Spatial Denoising -------------------------------
    TemporalSpatialDenoising(bufferDim, historyDim);

    // ---- debug -----
    if (0)
    {
        GpuErrorCheck(cudaDeviceSynchronize());
	    GpuErrorCheck(cudaPeekAtLastError());
    }

    // ------------------------------- post processing -------------------------------
    if (renderPassSettings.enablePostProcess)
    {
        // Histogram
        if (renderPassSettings.enableDownScalePasses)
        {
            DownScale4 <<<gridDim, blockDim>>>(
                GetBuffer2D(RenderColorBuffer),
                GetBuffer2D(ColorBuffer4),
                bufferDim);

            DownScale4 <<<gridDim4, blockDim>>>(
                GetBuffer2D(ColorBuffer4),
                GetBuffer2D(ColorBuffer16),
                bufferSize4);

            DownScale4 <<<gridDim16, blockDim>>>(
                GetBuffer2D(ColorBuffer16),
                GetBuffer2D(ColorBuffer64),
                bufferSize16);
        }

        if (renderPassSettings.enableHistogram)
            Histogram2<<<1, dim3(min(bufferSize64.x, 32), min(bufferSize64.y, 32), 1)>>>(
                /*out*/d_histogram,
                /*in*/GetBuffer2D(ColorBuffer64),
                bufferSize64);

        // Exposure
        if (renderPassSettings.enableAutoExposure)
        {
            AutoExposure<<<1, 1>>>(
                /*out*/d_exposure,
                /*in*/d_histogram,
                (float)(bufferSize64.x * bufferSize64.y),
                deltaTime);
        }
        else
        {
            float initExposureLum[4] = { postProcessParams.exposure, 1.0f, 1.0f, 1.0f }; // (exposureValue, historyAverageLuminance, historyBrightThresholdLuminance, unused)
	        GpuErrorCheck(cudaMemcpy(
                d_exposure,
                initExposureLum,
                4 * sizeof(float),
                cudaMemcpyHostToDevice));
        }

        // Bloom
        if (renderPassSettings.enableBloomEffect)
        {
            BloomGuassian<<<dim3(divRoundUp(bufferSize4.x, 12), divRoundUp(bufferSize4.y, 12), 1), dim3(16, 16, 1)>>>(
                GetBuffer2D(BloomBuffer4),
                GetBuffer2D(ColorBuffer4),
                bufferSize4,
                d_exposure);

            BloomGuassian<<<dim3(divRoundUp(bufferSize16.x, 12), divRoundUp(bufferSize16.y, 12), 1), dim3(16, 16, 1)>>>(
                GetBuffer2D(BloomBuffer16),
                GetBuffer2D(ColorBuffer16),
                bufferSize16,
                d_exposure);

            Bloom<<<gridDim, blockDim>>>(
                GetBuffer2D(RenderColorBuffer),
                GetBuffer2D(BloomBuffer4),
                GetBuffer2D(BloomBuffer16),
                bufferDim,
                bufferSize4,
                bufferSize16);
        }

        // Lens flare
        if (renderPassSettings.enableLensFlare)
        {
            if (sunPos.x > 0 && sunPos.x < 1 && sunPos.y > 0 && sunPos.y < 1 && sunDir.y > -0.0 && dot(sunDir, cbo.camera.dir) > 0)
            {
                sunPos -= Float2(0.5);
                sunPos.x *= (float)renderWidth / (float)renderHeight;
                LensFlarePred<<<1,1>>>(
                    GetBuffer2D(DepthBuffer),
                    sunPos,
                    sunUv,
                    GetBuffer2D(RenderColorBuffer),
                    bufferDim,
                    gridDim,
                    blockDim,
                    bufferDim);
            }
        }

        // Tone mapping
        if (renderPassSettings.enableToneMapping)
        {
            ToneMapping<<<gridDim, blockDim>>>(
                GetBuffer2D(RenderColorBuffer),
                bufferDim,
                d_exposure,
                postProcessParams);
        }
    }

    // Scale to final output
    BicubicScale<<<scaleGridDim, scaleBlockDim>>>(
        GetBuffer2D(ScaledColorBuffer),
        GetBuffer2D(RenderColorBuffer),
        outputDim,
        bufferDim);

    if (renderPassSettings.enablePostProcess)
    {
        // Sharpening
        if (renderPassSettings.enableSharpening)
        {
            SharpeningFilter<<<scaleGridDim, scaleBlockDim>>>(
                GetBuffer2D(ScaledColorBuffer),
                outputDim);
        }
    }

    // Output
    CopyToOutput<<<scaleGridDim, scaleBlockDim>>>(
        renderTarget,
        GetBuffer2D(ScaledColorBuffer),
        d_randGen,
        cbo,
        outputDim);

    // Debug
    #if DUMP_FRAME_NUM > 0
    if (cbo.frameNum == DUMP_FRAME_NUM || cbo.frameNum == DEBUG_FRAME)
    {
        // debug
	    CopyFrameBuffer <<<scaleGridDim, scaleBlockDim>>>(
            dumpFrameBuffer,
            renderTarget,
            outputDim);

        std::string outputName = "outputImage_frame_" + std::to_string(cbo.frameNum) + "_" + Timer::getTimeString() + ".ppm";

        writeToPPM(outputName, outputDim.x, outputDim.y, dumpFrameBuffer);
    }
    #endif
}