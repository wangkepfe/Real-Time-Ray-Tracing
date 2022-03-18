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
    constexpr float maxFpsAllowed = 75.0f;
    constexpr float minFrameTimeAllowed = 1000.0f / maxFpsAllowed;
    timer.updateWithLimiter(minFrameTimeAllowed);
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

    // Update params
    cbo.sampleParams = sampleParams;

    if (cbo.frameNum == 1)
    {
        // init history camera
        cbo.historyCamera.Setup(cbo.camera);
    }
}

__global__ void MeshDisplace(Float3* vertexBuffer, Float3* constVertexBuffer, Float3* normalBuffer, SurfObj heightMap, BlueNoiseRandGenerator randGen, uint size)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    Float3 vertex = constVertexBuffer[idx];
    Float3 normal = normalize(normalBuffer[idx]);

    //{
    //    Float4 randNum;
    //    randGen.idx       = idx;
    //    randGen.sampleIdx = 0;
    //    randNum = randGen.Rand4(0);

    //    Float3 normal = randNum.xyz;
    //    float offset = randNum.w;

    //    const float strength = 0.1f;
    //    vertex = vertex + normal * offset * strength;
    //}

    {
        Int2 texSize = Int2(1024, 1024);
        const float strength = 0.2f;
        const float uvScale = 0.5f;
        Float3 displaceX;
        Float3 displaceY;
        Float3 displaceZ;
        {
            Float2 uv(vertex.y, vertex.z);
            uv *= uvScale;

            float height = SampleNearest<Load2DFuncUshort1<float>, float, BoundaryFuncRepeat>(heightMap, uv, texSize);
            height -= 0.5f;

            displaceX = Float3(1, 0, 0) * height * strength;
        }
        {
            Float2 uv(vertex.x, vertex.z);
            uv *= uvScale;

            float height = SampleNearest<Load2DFuncUshort1<float>, float, BoundaryFuncRepeat>(heightMap, uv, texSize);
            height -= 0.5f;

            displaceY = Float3(0, 1, 0) * height * strength;
        }
        {
            Float2 uv(vertex.x, vertex.y);
            uv *= uvScale;

            float height = SampleNearest<Load2DFuncUshort1<float>, float, BoundaryFuncRepeat>(heightMap, uv, texSize);
            height -= 0.5f;

            displaceZ = Float3(0, 0, 1) * height * strength;
        }

        float wx = normal.x * normal.x;
        float wy = normal.y * normal.y;
        float wz = normal.z * normal.z;

        vertex = vertex + displaceX * wx + displaceY * wy + displaceZ * wz;
    }

    // if (blockIdx.x == gridDim.x * 0.5f)
    // {
    //     DEBUG_PRINT(uv);
    //     DEBUG_PRINT(height);
    // }

    vertexBuffer[idx] = vertex;
}

__device__ inline Float3 AtomicAdd3f(Float3& v, const Float3& a)
{
	Float3 old;
	old.x = atomicAdd(&v.x, a.x);
	old.y = atomicAdd(&v.y, a.y);
	old.z = atomicAdd(&v.z, a.z);
	return old;
}

__global__ void GenerateSmoothNormals(uint* indexBuffer, Float3* vertexBuffer, Float3* normalBuffer, uint triCount)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= triCount)
        return;

    uint index0 = indexBuffer[idx * 3 + 0];
    uint index1 = indexBuffer[idx * 3 + 1];
    uint index2 = indexBuffer[idx * 3 + 2];

    Float3 v0 = vertexBuffer[index0];
    Float3 v1 = vertexBuffer[index1];
    Float3 v2 = vertexBuffer[index2];

    // Float3 planeNormal = cross(v2 - v0, v1 - v0).normalize();
    // AtomicAdd3f(normalBuffer[index0], planeNormal);
    // AtomicAdd3f(normalBuffer[index1], planeNormal);
    // AtomicAdd3f(normalBuffer[index2], planeNormal);

    Float3 planeNormalMultArea = cross(v2 - v0, v2 - v1) / 2;

    float w0 = AngleBetween(v2 - v0, v1 - v0);
    float w1 = AngleBetween(v2 - v1, v0 - v1);
    float w2 = AngleBetween(v0 - v2, v1 - v2);

    AtomicAdd3f(normalBuffer[index0], planeNormalMultArea * w0);
    AtomicAdd3f(normalBuffer[index1], planeNormalMultArea * w1);
    AtomicAdd3f(normalBuffer[index2], planeNormalMultArea * w2);
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

        skyParams.needRegenerate = false;
    }

    GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

    // ------------------------------- Geometry processing -------------------------------
    if (cbo.frameNum == 1)
    {
        GenerateSmoothNormals <<< divRoundUp(triCountPadded, 64), 64 >>> (indexBuffer, constVertexBuffer, normalBuffer, triCountPadded);
        MeshDisplace <<< divRoundUp(numVertices, 64), 64 >>> (vertexBuffer, constVertexBuffer, normalBuffer, GetBuffer2D(SoilHeightBuffer), d_randGen, numVertices);
        GenerateSmoothNormals <<< divRoundUp(triCountPadded, 64), 64 >>> (indexBuffer, vertexBuffer, normalBuffer, triCountPadded);
    }

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
        GetBuffer2D(SkyBuffer),
        skyCdf,
        GetBuffer2D(SunBuffer),
        sunCdf,
        GetBuffer2D(MotionVectorBuffer),
        GetBuffer2D(NoiseLevelBuffer),
        GetBuffer2D(AlbedoBuffer),
        buffer2DManager.textures,
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