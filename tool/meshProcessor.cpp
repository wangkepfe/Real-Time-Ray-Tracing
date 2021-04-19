#include <iostream>
#include <vector>
#include <cassert>
#include <unordered_set>
#include <algorithm>
#include <fstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "kernel.cuh"

const std::string modelName = "resources/models/high-res-monkey.dae";
const std::string outputName = "resources/models/high-res-monkey.bin";

#define USE_60_BIT_MORTON_CODE 1

#if USE_60_BIT_MORTON_CODE
#define MORTON_TYPE uint64
#define MORTON_SCALE 1048575.f
#else
#define MORTON_TYPE uint
#define MORTON_SCALE 1023.f
#endif

struct AABB;
struct Triangle;

struct Morton
{
	MORTON_TYPE morton;
	uint order;
};

uint MortonCode3D(uint x, uint y, uint z) {
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8))  & 0x0300F00F;
	x = (x | (x << 4))  & 0x030C30C3;
	x = (x | (x << 2))  & 0x09249249;
	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y << 8))  & 0x0300F00F;
	y = (y | (y << 4))  & 0x030C30C3;
	y = (y | (y << 2))  & 0x09249249;
	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z << 8))  & 0x0300F00F;
	z = (z | (z << 4))  & 0x030C30C3;
	z = (z | (z << 2))  & 0x09249249;
	return x | (y << 1) | (z << 2);
}

uint64 MortonCode3D64(uint x, uint y, uint z) {
	uint loX = x & 1023u;
	uint loY = y & 1023u;
	uint loZ = z & 1023u;
	uint hiX = x >> 10u;
	uint hiY = y >> 10u;
	uint hiZ = z >> 10u;
	uint64 lo = MortonCode3D(loX, loY, loZ);
	uint64 hi = MortonCode3D(hiX, hiY, hiZ);
	return (hi << 30) | lo;
}

Float3 VecTypeConv(const aiVector3D& v)
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

			// push into list
			h_triangles.push_back(triangle);
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

AABB GetAabb(const Triangle& tri)
{
	AABB aabb(tri.v1, tri.v1);
	aabb.max = max3f(aabb.max, tri.v2);
	aabb.max = max3f(aabb.max, tri.v3);
	aabb.min = min3f(aabb.min, tri.v2);
	aabb.min = min3f(aabb.min, tri.v3);
	return aabb;
}

void CalculateAabb(std::vector<AABB>& aabbs, AABB& sceneAabb, const std::vector<Triangle>& triangles, uint size)
{
	for (uint i = 0; i < size; ++i)
	{
		AABB aabb = GetAabb(triangles[i]);
		aabbs[i] = aabb;
		sceneAabb.max = max3f(sceneAabb.max, aabb.max);
		sceneAabb.min = min3f(sceneAabb.min, aabb.min);
	}
	std::cout << "AABB generation done!\n";
}

void CalculateMortonOrder(std::vector<Morton>& morton, const AABB& sceneAabb, const std::vector<Triangle>& triangles, uint size)
{
	std::unordered_set<MORTON_TYPE> mortonSet;

	uint badCount = 0;
	for (uint i = 0; i < size; ++i)
	{
		Triangle triangle = triangles[i];
		Float3 center = (triangle.v1 + triangle.v2 + triangle.v3) / 3.0f;
		Float3 unitBox = (center - sceneAabb.min) / (sceneAabb.max - sceneAabb.min);

		MORTON_TYPE mortonCode = MortonCode3D64((uint)(unitBox.x * MORTON_SCALE),
								                (uint)(unitBox.y * MORTON_SCALE),
								                (uint)(unitBox.z * MORTON_SCALE));

		morton[i] = { mortonCode, i };

		auto it = mortonSet.find(mortonCode);
		if (it == mortonSet.end())
		{
			mortonSet.insert(mortonCode);
		}
		else
		{
			badCount++;
		}
	}
	std::cout << "The total identical morton code count is " << badCount << "\n";
	std::cout << "Morton code generation done!\n";
}

int main()
{
    // load triangles
    std::vector<Triangle> triangles;
    LoadScene(modelName.c_str(), triangles);
	uint trisize = triangles.size();

	// get aabbs
	std::vector<AABB> aabbs(trisize);
	AABB sceneAabb({ -FLT_MAX }, {FLT_MAX});
	CalculateAabb(aabbs, sceneAabb, triangles, trisize);

	// calculate morton order
	std::vector<Morton> morton(trisize);
	CalculateMortonOrder(morton, sceneAabb, triangles, trisize);

	// sort
	std::sort(morton.begin(), morton.end(), [](const Morton & lhs, const Morton & rhs) { return lhs.morton < rhs.morton; });
	std::cout << "Morton code sorting done!\n";

	// write out
	std::vector<Triangle> result(trisize);
	for (uint i = 0; i < trisize; ++i)
	{
		result[i] = triangles[morton[i].order];
	}

	std::ofstream outfile (outputName, std::ofstream::binary);
	if (outfile.good())
	{
		outfile.write(reinterpret_cast<const char*>(&trisize), sizeof(uint));
		outfile.write(reinterpret_cast<const char*>(result.data()), sizeof(Triangle) * trisize);
		outfile.close();
		std::cout << "Successfully write scene data to \"" << outputName << "\"!\n";
	} else {
		std::cout << "Error: Failed to write scene data to \"" << outputName << "\".\n";
	}

	return 0;
}