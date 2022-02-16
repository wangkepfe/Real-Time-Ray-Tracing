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

__host__ __device__ inline void Print(const char* name) { printf("%s\n", name); }
__host__ __device__ inline void Print(const char* name, const char* str) { printf("%s: %s\n", name, str); }
__host__ __device__ inline void Print(const char* name, const int& n) { printf("%s = %d\n", name, n); }
__host__ __device__ inline void Print(const char* name, const bool& n) { printf("%s = %s\n", name, n ? "true" : "false"); }
__host__ __device__ inline void Print(const char* name, const uint& n) { printf("%s = %d\n", name, n); }
__host__ __device__ inline void Print(const char* name, const Int2& n) { printf("%s = (%d, %d)\n", name, n.x, n.y); }
__host__ __device__ inline void Print(const char* name, const uint3& n) { printf("%s = (%d, %d, %d)\n", name, n.x, n.y, n.z); }
__host__ __device__ inline void Print(const char* name, const float& n) { printf("%s = %f\n", name, n); }
__host__ __device__ inline void Print(const char* name, const Float2& f3) { printf("%s = (%f, %f)\n", name, f3[0], f3[1]); }
__host__ __device__ inline void Print(const char* name, const Float3& f3) { printf("%s = (%f, %f, %f)\n", name, f3[0], f3[1], f3[2]); }
__host__ __device__ inline void Print(const char* name, const Float4& f4) { printf("%s = (%f, %f, %f, %f)\n", name, f4[0], f4[1], f4[2], f4[3]); }

#define NAN_DETECTER(__VALUE__) NanDetecter(__FILE__,__LINE__,#__VALUE__,__VALUE__);

__device__ inline bool isnan(const Float2& value) { return isnan(value.x) || isnan(value.y); }
__device__ inline bool isnan(const Float3& value) { return isnan(value.x) || isnan(value.y) || isnan(value.z); }
__device__ inline bool isnan(const Float4& value) { return isnan(value.x) || isnan(value.y) || isnan(value.z) || isnan(value.w); }

template<typename T>
__device__ inline void NanDetecter(const char* file, int line, const char* valueName, T& value)
{
	if (isnan(value))
    {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
        printf("%s:%d:%s:(%d,%d): Nan found! \n", file, line, valueName, x, y);
		value = T{};
    }
}

#define TEST_WITHIN_BOUND(__ARRAY__,__INDEX__,__SIZE__) TestWithinBound(__FILE__,__LINE__,#__ARRAY__, __INDEX__, __SIZE__);
#define SAFE_LOAD(__ARRAY__,__INDEX__,__SIZE__,__DEFAULT__) SafeLoad(__FILE__,__LINE__,#__ARRAY__,__ARRAY__,__INDEX__,__SIZE__,__DEFAULT__);

inline __host__ __device__ bool TestWithinBound(const char* file, int line, const char* arrayName, int idx, int maxSize)
{
	bool res = (idx >= 0) && (idx < maxSize);
	if (!res) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		printf("%s:%d:%s:(%d,%d): out of bound!\n", file, line, arrayName, x, y);
	}
	return res;
}

template<typename T>
inline __host__ __device__ T SafeLoad(const char* file, int line, const char* arrayName, const T* array, int idx, int maxSize, const T& defaultValue)
{
	if (TestWithinBound(file, line, arrayName, idx, maxSize)) {
		return array[idx];
	} else {
		return defaultValue;
	}
}