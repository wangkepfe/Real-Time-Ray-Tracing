#pragma once

#include "kernel.cuh"
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

inline Float3 VecTypeConv(const aiVector3D& v)
{
	return Float3(v.x, v.y, v.z);
}

void LoadSceneRecursive(std::vector<Triangle>& h_triangles, const aiScene* scene, const aiNode* node)
{
	// recursive for children
	for (uint childIdx = 0; childIdx < node->mNumChildren; childIdx++)
	{
		LoadSceneRecursive(h_triangles, scene, node->mChildren[childIdx]);
	}

	// traverse meshes of current node
	for (uint meshIdxIdx = 0; meshIdxIdx < node->mNumMeshes; meshIdxIdx++)
	{
		const uint meshIdx = node->mMeshes[meshIdxIdx];
		const aiMesh* mesh = scene->mMeshes[meshIdx];

		// traverse faces of current mesh
		for (uint faceIdx = 0; faceIdx < mesh->mNumFaces; faceIdx++)
		{
			// load three vertices
			const aiFace face = mesh->mFaces[faceIdx];
			assert(face.mNumIndices == 3);

			const aiVector3D v1 = mesh->mVertices[face.mIndices[0]];
			const aiVector3D v2 = mesh->mVertices[face.mIndices[1]];
			const aiVector3D v3 = mesh->mVertices[face.mIndices[2]];

			// constuct a triangle
			Triangle triangle(VecTypeConv(v1), VecTypeConv(v2), VecTypeConv(v3));

			// set normals
			// if (mesh->HasNormals())
			// {
			// 	triangle.n1 = VecTypeConv(mesh->mNormals[face.mIndices[0]]);
			// 	triangle.n2 = VecTypeConv(mesh->mNormals[face.mIndices[1]]);
			// 	triangle.n3 = VecTypeConv(mesh->mNormals[face.mIndices[2]]);
			// }

			// push into list
			if (h_triangles.size() < 1024 * 1024)
			{
				h_triangles.push_back(triangle);
			}
		}
	}
}

void LoadScene(const char* filePath, std::vector<Triangle>& h_triangles)
{
	// Open file and process
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(filePath, 0);

	// report error
	if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return;
    }
	else
	{
		std::cout << "Successfully loaded \"" << filePath << "\"!\n";
	}

	// load scene into triangle buffer
	LoadSceneRecursive(h_triangles, scene, scene->mRootNode);
}

cudaArray* LoadTextureRgba8(const char* texPath, cudaTextureObject_t& texObj)
{
	cudaArray* texArray;

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

	return texArray;
}

cudaArray* LoadTextureRgb8(const char* texPath, cudaTextureObject_t& texObj)
{
	cudaArray* texArray;

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

	return texArray;
}