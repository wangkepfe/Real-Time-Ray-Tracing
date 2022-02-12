#include "kernel.cuh"
#include "debugUtil.h"
#include "temporalDenoising.cuh"

void RayTracer::TemporalSpatialDenoising(Int2 bufferDim, Int2 historyDim)
{
    // Calculate Noise Level
    //
    //         |
    //         V
    //
    // Temporal Filter  <-----------------------------
    //
    //         |                                     ^
    //         V                                     |
    //
    // Spatial Filter 7x7  (Effective range 7x7)
    //
    //         |                                     ^
    //         V                                     |
    //
    // Copy To History Color Buffer ------------------
    //
    //         |
    //         V
    //
    // Calculate Noise Level
    //
    //         |
    //         V
    //
    // Spatial Filter Global 5x5, Stride=3 (Effective range 15x15)
    // Spatial Filter Global 5x5, Stride=6 (Effective range 30x30)
    // Spatial Filter Global 5x5, Stride=12 (Effective range 60x60)
    //
    //         |
    //         V
    //
    // Temporal Filter 2  <----------------
    //
    //         |                          ^
    //         V                          |
    //
    // Copy To History Color Buffer -------
    //
    // Done!


    UInt2 noiseLevel16x16Dim(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16));
    if (renderPassSettings.enableTemporalDenoising)
    {
        CalculateTileNoiseLevel<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 4, 1)>>>(
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer),
            bufferDim);

        TileNoiseLevel8x8to16x16<<<dim3(divRoundUp(noiseLevel16x16Dim.x, 8), divRoundUp(noiseLevel16x16Dim.y, 8), 1), dim3(8, 8, 1)>>>(
            GetBuffer2D(NoiseLevelBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16));

        if (cbo.frameNum != 1)
        {
            TemporalFilter <<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 8, 1)>>>(
                cbo,
                GetBuffer2D(RenderColorBuffer),
                GetBuffer2D(AccumulationColorBuffer),
                GetBuffer2D(NormalBuffer),
                GetBuffer2D(DepthBuffer),
                GetBuffer2D(HistoryDepthBuffer),
                GetBuffer2D(MotionVectorBuffer),
                GetBuffer2D(NoiseLevelBuffer),
                bufferDim, historyDim);
        }
    }

    if (renderPassSettings.enableLocalSpatialFilter)
    {
        SpatialFilter7x7<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            bufferDim);
    }

    if (renderPassSettings.enableTemporalDenoising)
    {
        CopyToHistoryColorBuffer<<<gridDim, blockDim>>>(
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(AccumulationColorBuffer),
            bufferDim);
    }

    CalculateTileNoiseLevel<<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 4, 1)>>>(
        GetBuffer2D(RenderColorBuffer),
        GetBuffer2D(DepthBuffer),
        GetBuffer2D(NoiseLevelBuffer),
        bufferDim);

    TileNoiseLevel8x8to16x16<<<dim3(divRoundUp(noiseLevel16x16Dim.x, 8), divRoundUp(noiseLevel16x16Dim.y, 8), 1), dim3(8, 8, 1)>>>(
        GetBuffer2D(NoiseLevelBuffer),
        GetBuffer2D(NoiseLevelBuffer16x16));

    if (renderPassSettings.enableNoiseLevelVisualize)
    {
        TileNoiseLevelVisualize<<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            bufferDim);
    }

    if (renderPassSettings.enableWideSpatialFilter)
    {
        SpatialFilterGlobal5x5<3><<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            bufferDim);

        SpatialFilterGlobal5x5<6><<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            bufferDim);

        SpatialFilterGlobal5x5<12><<<dim3(divRoundUp(renderWidth, 16), divRoundUp(renderHeight, 16), 1), dim3(16, 16, 1)>>>(
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            bufferDim);
    }

    if (renderPassSettings.enableTemporalDenoising2)
    {
        if (cbo.frameNum != 1)
        {
            TemporalFilter2 <<<dim3(divRoundUp(renderWidth, 8), divRoundUp(renderHeight, 8), 1), dim3(8, 8, 1)>>>(
                cbo,
                GetBuffer2D(RenderColorBuffer), GetBuffer2D(HistoryColorBuffer),
                GetBuffer2D(NormalBuffer),
                GetBuffer2D(DepthBuffer), GetBuffer2D(HistoryDepthBuffer),
                GetBuffer2D(MotionVectorBuffer),
                GetBuffer2D(NoiseLevelBuffer),
                bufferDim, historyDim);
        }
        CopyToHistoryColorDepthBuffer<<<gridDim, blockDim>>>(
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(HistoryColorBuffer),
            GetBuffer2D(HistoryDepthBuffer),
            bufferDim);
    }
}