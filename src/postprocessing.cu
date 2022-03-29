#include "kernel.cuh"
#include "debugUtil.h"
#include "postprocessing.cuh"

void RayTracer::PostProcessing(Int2 bufferDim, Int2 outputDim)
{
    #if USE_PRECALCULATED_GAUSSIAN == 0
    if (cbo.frameNum == 1)
    {
        CalculateGaussian3x3();
        CalculateGaussian5x5();
        CalculateGaussian7x7();
    }
    #endif

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
                deltaTime,
                postProcessParams.gain);
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

        // Tone mapping
        if (renderPassSettings.enableToneMapping)
        {
            if (postProcessParams.toneMappingType == ToneMappingType::Uncharted)
            {
                ToneMappingUncharted<<<scaleGridDim, scaleBlockDim>>>(
                    GetBuffer2D(ScaledColorBuffer),
                    outputDim,
                    d_exposure,
                    postProcessParams);
            }
            else if (postProcessParams.toneMappingType == ToneMappingType::ACES1)
            {
                ToneMappingACES<<<scaleGridDim, scaleBlockDim>>>(
                    GetBuffer2D(ScaledColorBuffer),
                    outputDim,
                    d_exposure,
                    postProcessParams);
            }
            else if (postProcessParams.toneMappingType == ToneMappingType::ACES2)
            {
                ToneMappingACES2<<<scaleGridDim, scaleBlockDim>>>(
                    GetBuffer2D(ScaledColorBuffer),
                    outputDim,
                    d_exposure,
                    postProcessParams);
            }
            else if (postProcessParams.toneMappingType == ToneMappingType::Reinhard)
            {
                ToneMappingReinhardExtended<<<scaleGridDim, scaleBlockDim>>>(
                    GetBuffer2D(ScaledColorBuffer),
                    outputDim,
                    d_exposure,
                    postProcessParams);
            }
        }
    }
}