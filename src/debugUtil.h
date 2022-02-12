#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <cassert>

#define LETS_DEBUG 1
#if LETS_DEBUG
#define IS_DEBUG_PIXEL() blockIdx.x * blockDim.x + threadIdx.x == gridDim.x * blockDim.x * 0.5 && blockIdx.y * blockDim.y + threadIdx.y == gridDim.y * blockDim.y * 0.5
#else
#define IS_DEBUG_PIXEL() 0
#endif
#define DEBUG_PRINT(__THING__) if(IS_DEBUG_PIXEL()) { Print(#__THING__, __THING__); }
#define DEBUG_PRINT_STRING(__STRING__) if(IS_DEBUG_PIXEL()) { Print(__STRING__); }
#define DEBUG_PRINT_BAR if(IS_DEBUG_PIXEL()) { Print("------------------------------"); }
#define NAN_DETECTER(__THING__) NanDetecter(__FUNCTION__,#__THING__, __THING__);

#define DEBUG_CUDA() GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

static const std::string logDir = "C:/Ultimate-Realism-Renderer/log/";

__device__ inline bool IsPixelAt(float u, float v)
{
	int centerx = gridDim.x * blockDim.x * u;
	int centery = gridDim.y * blockDim.y * v;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	return (x == centerx) && (y == centery);
}

__device__ inline bool IsCenterBlock()
{
	int x = gridDim.x * 0.5;
	int y = gridDim.y * 0.5;
	return (blockIdx.x == x) && (blockIdx.y == y);
}

__device__ inline bool IsFirstBlock()
{
	int x = 1;
	int y = 0;
	return (blockIdx.x == x) && (blockIdx.y == y);
}

struct Triangle;
struct AABB;
struct BVHNode;

namespace std
{
	inline ostream& operator<<(ostream& os, const Float3& vec)
	{
		return os << vec.x << "," << vec.y << "," << vec.z << ",";
	}

	inline ostream& operator<<(ostream& os, const Triangle& triangle)
	{
		return os << triangle.v1 << triangle.v2 << triangle.v3;
	}

	inline ostream& operator<<(ostream& os, const AABB& aabb)
	{
		return os << aabb.max << aabb.min;
	}

	inline ostream& operator<<(ostream& os, const BVHNode& bvhNode)
	{
		return os << bvhNode.aabb.GetLeftAABB() << bvhNode.aabb.GetRightAABB()
		          << bvhNode.idxLeft << ","
				  << bvhNode.idxRight << ","
				  << bvhNode.isLeftLeaf << ","
				  << bvhNode.isRightLeaf << ",";
	}
};

inline void writeToPPM(const std::string& filename, int width, int height, uchar4* devicePtr)
{
	uchar4* hostPtr = new uchar4[width * height];
	assert(hostPtr != nullptr);

	GpuErrorCheck(cudaMemcpy(hostPtr, devicePtr, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));

	std::string fullFilename = logDir + filename;

	// std::cout << fullFilename << "\n";

    FILE *f = fopen(fullFilename.c_str(), "w");
	assert(f != nullptr);

    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);

    for (int i = 0; i < width * height; i++)
	{
        fprintf(f, "%d %d %d ", hostPtr[i].x, hostPtr[i].y, hostPtr[i].z);
	}

    fclose(f);
    printf("Successfully wrote result image to %s\n", filename.c_str());

	delete hostPtr;
}

template<typename T>
inline void DebugPrintFile(const std::string& filename, T* devicePtr, uint arraySize)
{
	T* hostPtr = new T[arraySize];
	assert(hostPtr != nullptr);

	GpuErrorCheck(cudaMemcpy(hostPtr, devicePtr, arraySize * sizeof(T), cudaMemcpyDeviceToHost));

	std::string fullFilename = logDir + filename;

	std::ofstream myfile;
	myfile.open (fullFilename);

	if (myfile.is_open())
	{
		for (int i = 0; i < arraySize; ++i)
		{
			myfile << hostPtr[i] << "\n";
		}

		myfile.close();
	}

	delete hostPtr;
}

__device__ inline void Print(const char* name) { printf("%s\n", name); }
__device__ inline void Print(const char* name, const int& n) { printf("%s = %d\n", name, n); }
__device__ inline void Print(const char* name, const bool& n) { printf("%s = %s\n", name, n ? "true" : "false"); }
__device__ inline void Print(const char* name, const uint& n) { printf("%s = %d\n", name, n); }
__device__ inline void Print(const char* name, const Int2& n) { printf("%s = (%d, %d)\n", name, n.x, n.y); }
__device__ inline void Print(const char* name, const uint3& n) { printf("%s = (%d, %d, %d)\n", name, n.x, n.y, n.z); }
__device__ inline void Print(const char* name, const float& n) { printf("%s = %f\n", name, n); }
__device__ inline void Print(const char* name, const Float2& f3) { printf("%s = (%f, %f)\n", name, f3[0], f3[1]); }
__device__ inline void Print(const char* name, const Float3& f3) { printf("%s = (%f, %f, %f)\n", name, f3[0], f3[1], f3[2]); }
__device__ inline void Print(const char* name, const Float4& f4) { printf("%s = (%f, %f, %f, %f)\n", name, f4[0], f4[1], f4[2], f4[3]); }

__device__ inline void NanDetecter(const char* name1, const char* name2, float& v)
{
	if (isnan(v))
    {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
        //printf("%s::%s: Nan found at (%d, %d)!\n", name1, name2, x, y);
		v = 0;
    }
}

__device__ inline void NanDetecter(const char* name1, const char* name2, Float2& v2)
{
	if (isnan(v2.x) || isnan(v2.y))
    {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
        //printf("%s::%s: Nan found at (%d, %d)!\n", name1, name2, x, y);
		v2 = 0;
    }
}

__device__ inline void NanDetecter(const char* name1, const char* name2, Float3& v3)
{
	if (isnan(v3.x) || isnan(v3.y) || isnan(v3.z))
    {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
        //printf("%s::%s: Nan found at (%d, %d)!\n", name1, name2, x, y);
		v3 = 0;
    }
}