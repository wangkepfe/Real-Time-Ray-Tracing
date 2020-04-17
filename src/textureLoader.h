#pragma once

#include "kernel.cuh"
#include <HDRloader.h>

//// HDR environment map
//void RayTracer::InitHdrEnvMap()
//{
//	HDRImage HDRresult;
//
//	if (HDRLoader::load(HDRmapname, HDRresult))
//		printf("HDR environment map loaded. Width: %d Height: %d\n", HDRresult.width, HDRresult.height);
//	else
//	{
//		printf("HDR environment map not found\nAn HDR map is required as light source. Exiting now...\n");
//		system("PAUSE");
//		exit(0);
//	}
//
//	int HDRwidth = HDRresult.width;
//	int HDRheight = HDRresult.height;
//	Float4 * cpuHDRenv = new Float4[HDRwidth * HDRheight];
//
//	for (int i = 0; i < HDRwidth; i++)
//	{
//		for (int j = 0; j < HDRheight; j++)
//		{
//			int idx = 3 * (HDRwidth * j + i);
//			int idx2 = HDRwidth * (j) + i;
//			cpuHDRenv[idx2] = Float4(HDRresult.colors[idx], HDRresult.colors[idx + 1], HDRresult.colors[idx + 2], 0.0f);
//		}
//	}
//
//	// copy HDR map to CUDA
//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
//	cudaMallocArray(&d_HdrEnvMap, &channelDesc, HDRwidth, HDRheight);
//    cudaMemcpyToArray(d_HdrEnvMap, 0, 0, cpuHDRenv, HDRwidth * HDRheight * sizeof(float4), cudaMemcpyHostToDevice);
//
//	delete cpuHDRenv;
//}