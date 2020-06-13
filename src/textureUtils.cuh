#pragma once

#include "kernel.cuh"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void LoadTextureRgba8(const char* texPath, cudaArray* texArray, cudaTextureObject_t& texObj)
{
	// stbi image load
	int texWidth, texHeight, texChannel;
	unsigned char* buffer = stbi_load(texPath, &texWidth, &texHeight, &texChannel, STBI_rgb_alpha);
	std::cout << "Texture loaded: " << texPath << ", width=" << texWidth << ", height=" << texHeight << ", channel=" << texChannel << "\n";
	assert(buffer != NULL);

	// copy to cuda format
	uchar4* cpuTextureBuffer = new uchar4[texWidth * texHeight];
	for (int i = 0; i < texWidth * texHeight; ++i)
	{
		cpuTextureBuffer[i] = make_uchar4(buffer[i * 4], buffer[i * 4 + 1], buffer[i * 4 + 2], buffer[i * 4 + 3]);
	}

	// channel description
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&texArray, &channelDesc, texWidth, texHeight);
	cudaMemcpyToArray(texArray, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(uchar4), cudaMemcpyHostToDevice);

	// resource description
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = texArray;

	// texture description
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeWrap;
	texDesc.addressMode[1]   = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// create
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	assert(texObj != NULL);

	// free cpu buffer
	delete cpuTextureBuffer;
	stbi_image_free(buffer);
}

void LoadTextureRgb8(const char* texPath, cudaArray* texArray, cudaTextureObject_t& texObj)
{
	// stbi image load
	int texWidth, texHeight, texChannel;
	unsigned char* buffer = stbi_load(texPath, &texWidth, &texHeight, &texChannel, STBI_rgb);
	std::cout << "Texture loaded: " << texPath << ", width=" << texWidth << ", height=" << texHeight << ", channel=" << texChannel << "\n";
	assert(buffer != NULL);

	// copy to cuda format
	uchar4* cpuTextureBuffer = new uchar4[texWidth * texHeight];
	for (int i = 0; i < texWidth * texHeight; ++i)
	{
		cpuTextureBuffer[i] = make_uchar4(buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2], 0);
	}

	// channel description
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&texArray, &channelDesc, texWidth, texHeight);
	cudaMemcpyToArray(texArray, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(uchar4), cudaMemcpyHostToDevice);

	// resource description
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = texArray;

	// texture description
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeWrap;
	texDesc.addressMode[1]   = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// create
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	assert(texObj != NULL);

	// free cpu buffer
	delete cpuTextureBuffer;
	stbi_image_free(buffer);
}