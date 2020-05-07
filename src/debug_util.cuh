#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__device__ bool IsCenterPixel()
{
	int centerx = gridDim.x * blockDim.x * 0.12;
	int centery = gridDim.y * blockDim.y * 0.6;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	return (x == centerx) && (y == centery);
}

__device__ bool IsCenterBlock()
{
	int x = gridDim.x * 0.5;
	int y = gridDim.y * 0.55;
	return (blockIdx.x == x) && (blockIdx.y == y);
}

__device__ bool IsFirstBlock()
{
	int x = 1;
	int y = 0;
	return (blockIdx.x == x) && (blockIdx.y == y);
}

__device__ void DebugPrint(const char* name)
{
	if (IsCenterBlock())
	{
		printf("%s\n", name);
	}
}

__device__ void DebugPrint(const char* name, int num)
{
	if (IsCenterBlock())
	{
		printf("%s = %d\n", name, num);
	}
}

__device__ void DebugPrint(const char* name, uint num)
{
	if (IsCenterPixel())
	{
		printf("%s = %d\n", name, num);
	}
}

__device__ void DebugPrint(const char* name, unsigned long long int mask)
{
	if (IsCenterBlock())
	{
		printf("%20s = 0x%16llx\n", name, mask);
	}
}

__device__ void DebugPrint(const char* name, float num)
{
	if (IsCenterBlock())
	{
		printf("%s = %f\n", name, num);
	}
}

__device__ void DebugPrint(const char* name, const Float3& vec)
{
	if (IsCenterBlock())
	{
		printf("%s = (%f, %f, %f)\n", name, vec[0], vec[1], vec[2]);
	}
}

__device__ void DebugPrint(const char* name, const Float4& vec)
{
	if (IsCenterBlock())
	{
		printf("%s = (%f, %f, %f, %f)\n", name, vec[0], vec[1], vec[2], vec[3]);
	}
}

__device__ void DebugPrint(const char* name, const Int2& vec)
{
	if (IsCenterBlock())
	{
		printf("%s = (%d, %d)\n", name, vec[0], vec[1]);
	}
}

__device__ void Print(const char* name, const int& n) { printf("%s = %d\n", name, n); }
__device__ void Print(const char* name, const bool& n) { printf("%s = %s\n", name, n ? "true" : "false"); }
__device__ void Print(const char* name, const uint& n) { printf("%s = %d\n", name, n); }
__device__ void Print(const char* name, const uint3& n) { printf("%s = (%d, %d, %d)\n", name, n.x, n.y, n.z); }
__device__ void Print(const char* name, const float& n) { printf("%s = %f\n", name, n); }
__device__ void Print(const char* name, const Float3& f3) { printf("%s = (%f, %f, %f)\n", name, f3[0], f3[1], f3[2]); }