#pragma once

#include "kernel.cuh"
#include "sampler.cuh"
#include "common.cuh"
#include "bsdf.cuh"

extern __constant__ float filterKernel[25];

// __global__ void SpatialFilterReconstruction5x5(
// 	SurfObj colorBuffer,
// 	SurfObj indirectLightColorBuffer,
//     SceneMaterial sceneMaterial,
// 	Int2    size)
// {
// 	int x = threadIdx.x + blockIdx.x * 16;
// 	int y = threadIdx.y + blockIdx.y * 16;

// 	ushort mask = Load2DHalf3Ushort1(colorBuffer, Int2(x, y)).w;

// 	struct LDS
// 	{
// 		uint4 reconInfo;
// 		// Float3 color;
// 	};
// 	__shared__ LDS sharedBuffer[20 * 20];

// 	// calculate address
// 	int id = (threadIdx.x + threadIdx.y * 16);

// 	int x1 = blockIdx.x * 16 - 2 + id % 20;
// 	int y1 = blockIdx.y * 16 - 2 + id / 20;

// 	int x2 = blockIdx.x * 16 - 2 + (id + 256) % 20;
// 	int y2 = blockIdx.y * 16 - 2 + (id + 256) / 20;

// 	// global load 1
// 	// Float3 colorValue1  = Load2DHalf3Ushort1(colorBuffer, Int2(x1, y1)).xyz;
// 	uint4 reconInfo1 = Load2D<uint4>(indirectLightColorBuffer, Int2(x1, y1)).xyz;

// 	// store to lds 1
// 	sharedBuffer[id      ] = {
// 	//	colorValue1,
// 		reconInfo1 };

// 	if (id + 256 < 400)
// 	{
// 		// global load 2
// 		// Float3 colorValue2  = Load2DHalf3Ushort1(colorBuffer, Int2(x2, y2)).xyz;
// 		uint4 reconInfo2 = Load2D<uint4>(indirectLightColorBuffer, Int2(x2, y2)).xyz;

// 		// store to lds 2
// 		sharedBuffer[id + 256] = {
// 		//	colorValue2,
// 			reconInfo2 };
// 	}

// 	__syncthreads();

// 	if (x >= size.x && y >= size.y) return;

// 	// load center
// 	Float3 colorCenter = Load2DHalf3Ushort1(colorBuffer, Int2(x, y)).xyz;

// 	LDS center = sharedBuffer[threadIdx.x + 2 + (threadIdx.y + 2) * 20];
// 	ReconstructionInfo reconInfoCenter(center.reconInfo);
// 	ushort matId;
// 	reconInfoCenter.unpackMatId(matId);
//     SurfaceMaterial mat = sceneMaterial.materials[matId];

// 	if (matId == 0xffff || matId == MAT_SKY) return;

// 	// -------------------------------- atrous filter --------------------------------
// 	Float3 sumOfColor = 0;
// 	Float3 sumOfWeight = 0;

// 	#pragma unroll
// 	for (int i = 0; i < 25; i += 1)
// 	{
// 		int xoffset = i % 5;
// 		int yoffset = i / 5;

// 		LDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * 20];

// 		ReconstructionInfo reconInfo(bufferReadTmp.reconInfo);
// 		float cosThetaWoWh, cosThetaWo,  cosThetaWi,  cosThetaWh;

//         Float3 beta;
// 		if (matId == LAMBERTIAN_DIFFUSE)
// 		{
//             beta = LambertianBsdfOverPdf(mat.albedo);
// 		}
//         else if (matId == MICROFACET_REFLECTION)
//         {
//             reconInfo.unpackCosines(cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh);
//             Float3 brdf;
//             float pdf;
//             MacrofacetReflection2(
//                 beta, brdf, pdf,
//                 mat.F0, mat.albedo, mat.alpha,
//                 cosThetaWoWh, cosThetaWo, cosThetaWi, cosThetaWh);
//         }
//         else
//         {
//             continue;
//         }

//         Float3 Li;
//         reconInfo.unpackLi(Li);
//         Float3 color = beta * Li;
// 		Float3 weight = 1.0f;

// 		// accumulate
// 		sumOfColor  += color * weight;
// 		sumOfWeight += weight;
// 	}

// 	// final color
// 	Float3 finalColor = SafeDivide3f(sumOfColor, sumOfWeight);

// 	if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
//     {
//         printf("SpatialFilterReconstruction5x5: nan found at (%d, %d)\n", x, y);
//         finalColor = 0;
//     }

// 	// store to current
// 	Store2DHalf3Ushort1( { finalColor, mask } , colorBuffer, Int2(x, y));
// }
