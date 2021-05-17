#pragma once

#include "kernel.cuh"
#include "sampler.cuh"
#include "common.cuh"
#include "bsdf.cuh"

extern __constant__ float cGaussian5x5[25];

__global__ void SpatialReconstruction5x5(
	SurfObj                colorBuffer,
    SurfObj                normalbuffer,
    SurfObj                depthbuffer,
	SurfObj                LiBuffer,
    SurfObj                wiBuffer,
    SceneMaterial          sceneMaterial,
    BlueNoiseRandGenerator randGen,
    ConstBuffer            cbo,
	Int2                   size)
{
	int x = threadIdx.x + blockIdx.x * 16;
	int y = threadIdx.y + blockIdx.y * 16;

    // shared buffer
	struct LDS
	{
        Half3 Li;
		Half3 wi;
        half depth;
	};
	__shared__ LDS sharedBuffer[20 * 20];

    // global read coordinates
	int id = (threadIdx.x + threadIdx.y * 16);
	int x1 = blockIdx.x * 16 - 2 + id % 20;
	int y1 = blockIdx.y * 16 - 2 + id / 20;
	int x2 = blockIdx.x * 16 - 2 + (id + 256) % 20;
	int y2 = blockIdx.y * 16 - 2 + (id + 256) / 20;

    // global read 1
	sharedBuffer[id] =
    {
        Load2DHalf4<Half3>(LiBuffer, Int2(x1, y1)),
        Load2DHalf4<Half3>(wiBuffer, Int2(x1, y1)),
        Load2DHalf1<half>(depthbuffer, Int2(x1, y1))
    };

    // global read 2
	if (id + 256 < 400)
	{
		sharedBuffer[id + 256] =
        {
            Load2DHalf4<Half3>(LiBuffer, Int2(x2, y2)),
            Load2DHalf4<Half3>(wiBuffer, Int2(x2, y2)),
			Load2DHalf1<half>(depthbuffer, Int2(x2, y2))
        };
	}

	__syncthreads();

    // early return by size
	if (x >= size.x && y >= size.y) return;

    Float3Ushort1 colorLoad = Load2DHalf3Ushort1(colorBuffer, Int2(x, y));
	Float3 colorCenter = colorLoad.xyz;
    ushort matId = colorLoad.w;
    SurfaceMaterial mat = sceneMaterial.materials[matId];
    uint matType = mat.type;

    // early return by material type
	if ((matType == LAMBERTIAN_DIFFUSE || matType == MICROFACET_REFLECTION) == false) return;

    // normal and depth
    Float3 normalValue = Load2DHalf4ToFloat3(normalbuffer, Int2(x, y));

    int ldsCenterLocation = threadIdx.x + 2 + (threadIdx.y + 2) * 20;
    float depthValue = __half2float(sharedBuffer[ldsCenterLocation].depth);

    // rand gen (three loads for each rand number)
    randGen.idx       = Int2(x, y);
    int sampleNum     = 1;
    int sampleIdx     = 0;
    randGen.sampleIdx = cbo.frameNum * sampleNum + sampleIdx;
    Float2 randNum     = randGen.Rand2(0);

    // generate ray
    Float2 sampleUv;
    Float3 rayOrig, rayDir;
    GenerateRay(rayOrig, rayDir, sampleUv, cbo.camera, Int2(x, y), Float2(randNum.x, randNum.y), Float2(randNum.x, randNum.y));

	// -------------------------------- atrous filter --------------------------------
	Float3 sumOfColor = 0;
	float sumOfWeight = 0;

    const float sigma_depth = 4.0f;

	for (int i = 0; i < 25; i += 1)
	{
		int xoffset = i % 5;
		int yoffset = i / 5;

		LDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * 20];

        float weight = 1;
        float depth = __half2float(bufferReadTmp.depth);
        float deltaDepth = (depthValue - depth) / sigma_depth;
		deltaDepth = deltaDepth * deltaDepth;
        float depthWeight = expf(-0.5f * deltaDepth);

        if (depthWeight < 0.1f)
        {
            continue;
        }

        Float3 beta;
		if (matType == LAMBERTIAN_DIFFUSE)
		{
            beta = LambertianBsdfOverPdf(mat.albedo);
		}
        else if (matType == MICROFACET_REFLECTION)
        {
			Float3 wi = half3ToFloat3(bufferReadTmp.wi).normalized();

            Float3 brdf;
            float pdf;

            MacrofacetReflection(
                beta, brdf, pdf,
                normalValue, -rayDir, wi,
                mat.F0, mat.albedo, mat.alpha);

            beta = clamp3f(beta, 0, 1);
        }
        else
        {
            continue;
        }

        Float3 Li = half3ToFloat3(bufferReadTmp.Li);
        Float3 color = beta * Li;
        weight *= depthWeight;

        weight *= cGaussian5x5[xoffset + yoffset * 5];

		// accumulate
		sumOfColor  += color * weight;
		sumOfWeight += weight;
	}

	// final color
	Float3 finalColor;

    if (sumOfWeight == 0)
    {
        finalColor = 0;
    }
    else
    {
        finalColor = sumOfColor / sumOfWeight;
    }

	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        printf("SpatialReconstruction5x5: nan found at (%d, %d)\n", x, y);
        finalColor = 0;
    }

	// store to current
	Store2DHalf3Ushort1( { finalColor, matId } , colorBuffer, Int2(x, y));
}