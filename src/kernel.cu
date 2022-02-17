#include "kernel.cuh"
#include "debugUtil.h"
#include "scan.cuh"
#include "sky.cuh"
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

__global__ void CopyToOutput(
	SurfObj* renderTarget,
	SurfObj  outputResColor,
	BlueNoiseRandGenerator randGen,
	ConstBuffer            cbo,
	Int2     outSize)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx.x >= outSize.x || idx.y >= outSize.y) return;

	Float3 sampledColor = Load2DHalf4(outputResColor, idx).xyz;

	 // setup rand gen
    randGen.idx       = idx;
    int sampleNum     = 1;
    int sampleIdx     = 0;
    randGen.sampleIdx = cbo.frameNum * sampleNum + sampleIdx;
    Float4 blueNoise  = randGen.Rand4(0);

	Float3 jitter = blueNoise.xyz / 256 - 1 / 512;

	sampledColor = clamp3f(sampledColor + jitter, Float3(0), Float3(1) - MachineEpsilon());

	surf2Dwrite(make_uchar4(sampledColor.x * 256,
							sampledColor.y * 256,
							sampledColor.z * 256,
							1.0f),
		        renderTarget[0],
				idx.x * 4,
				idx.y);
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

        float targetFrameTimeHigh = 1000.0f / (g_settings->targetFps - 2);
		float targetFrameTimeLow = 1000.0f / (g_settings->targetFps + 2);

        if (targetFrameTimeHigh < deltaTime || targetFrameTimeLow > deltaTime) {
            float ratio = (1000.0f / g_settings->targetFps) / deltaTime;
            ratio = sqrtf(ratio);
            renderWidth *= ratio;
        }

        // Safe resolution
        renderWidth = renderWidth + ((renderWidth % 16 < 8) ? (- renderWidth % 16) : (16 - renderWidth % 16));
        renderWidth = clampi(renderWidth, minRenderWidth, maxRenderWidth);
        renderHeight = (renderWidth / 16) * 9;

        cbo.camera.resolution = Float2(renderWidth, renderHeight);
        cbo.camera.update();

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

    GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

    // ------------------------------- Sky -------------------------------
    if (cbo.frameNum == 1)
    {
        InitSkyConstantBuffer();
    }
    if (skyParams.needRegenerate)
    {
        skyParams.sunScalar = max(skyParams.sunScalar, 0.00001f);
        skyParams.skyScalar = max(skyParams.skyScalar, 0.00001f);
        skyParams.sunAngle = max(skyParams.sunAngle, 0.51f);

        UpdateSkyState(sunDir);

        Sky<<<dim3(SKY_WIDTH / 8, SKY_HEIGHT / 8, 1), dim3(8, 8, 1)>>>(GetBuffer2D(SkyBuffer), skyPdf, Int2(SKY_WIDTH, SKY_HEIGHT), sunDir, skyParams);
        Scan(skyPdf, skyCdf, skyCdfScanTmp, SKY_SIZE, SKY_SCAN_BLOCK_SIZE, 1);

        SkySun<<<dim3(SUN_WIDTH / 8, SUN_HEIGHT / 8, 1), dim3(8, 8, 1)>>>(GetBuffer2D(SunBuffer), sunPdf, Int2(SUN_WIDTH, SUN_HEIGHT), sunDir, skyParams);
        Scan(sunPdf, sunCdf, sunCdfScanTmp, SUN_SIZE, SUN_SCAN_BLOCK_SIZE, 1);

        cbo.sunAngleCosThetaMax = cosf(skyParams.sunAngle * M_PI / 180.0f / 2.0f);
        cbo.sampleSkyVsSun = skyParams.sampleSkyVsSun;

        skyParams.needRegenerate = false;
    }

    GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

    // ------------------------------- BVH -------------------------------
    BuildBvhLevel1();
    BuildBvhLevel2();

    GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

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
        GetBuffer2D(SunBuffer),
        sunCdf,
        GetBuffer2D(MotionVectorBuffer),
        GetBuffer2D(NoiseLevelBuffer),
        bufferDim);

    GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

    // update history camera
    cbo.historyCamera.Setup(cbo.camera);

    // ------------------------------- Temporal Spatial Denoising -------------------------------
    TemporalSpatialDenoising(bufferDim, historyDim);

    GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

    // ------------------------------- post processing -------------------------------
    PostProcessing(bufferDim, outputDim);

    GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

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

    if (1)
    {
        GpuErrorCheck(cudaDeviceSynchronize());
	    GpuErrorCheck(cudaPeekAtLastError());
    }
}