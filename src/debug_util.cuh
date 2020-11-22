#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define LETS_DEBUG 1
#if LETS_DEBUG
#define IS_DEBUG_PIXEL() blockIdx.x * blockDim.x + threadIdx.x == gridDim.x * blockDim.x * 0.5 && blockIdx.y * blockDim.y + threadIdx.y == gridDim.y * blockDim.y * 0.5
#else
#define IS_DEBUG_PIXEL() 0
#endif
#define DEBUG_PRINT(__THING__) if(IS_DEBUG_PIXEL()) { Print(#__THING__, __THING__); }
#define DEBUG_PRINT_BAR if(IS_DEBUG_PIXEL()) { Print("------------------------------"); }

#define DEBUG_CUDA() GpuErrorCheck(cudaDeviceSynchronize()); GpuErrorCheck(cudaPeekAtLastError());

__device__ bool IsPixelAt(float u, float v)
{
	int centerx = gridDim.x * blockDim.x * u;
	int centery = gridDim.y * blockDim.y * v;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	return (x == centerx) && (y == centery);
}

__device__ bool IsCenterBlock()
{
	int x = gridDim.x * 0.5;
	int y = gridDim.y * 0.5;
	return (blockIdx.x == x) && (blockIdx.y == y);
}

__device__ bool IsFirstBlock()
{
	int x = 1;
	int y = 0;
	return (blockIdx.x == x) && (blockIdx.y == y);
}

__device__ void Print(const char* name) { printf("%s\n", name); }
__device__ void Print(const char* name, const int& n) { printf("%s = %d\n", name, n); }
__device__ void Print(const char* name, const bool& n) { printf("%s = %s\n", name, n ? "true" : "false"); }
__device__ void Print(const char* name, const uint& n) { printf("%s = %d\n", name, n); }
__device__ void Print(const char* name, const Int2& n) { printf("%s = (%d, %d)\n", name, n.x, n.y); }
__device__ void Print(const char* name, const uint3& n) { printf("%s = (%d, %d, %d)\n", name, n.x, n.y, n.z); }
__device__ void Print(const char* name, const float& n) { printf("%s = %f\n", name, n); }
__device__ void Print(const char* name, const Float2& f3) { printf("%s = (%f, %f)\n", name, f3[0], f3[1]); }
__device__ void Print(const char* name, const Float3& f3) { printf("%s = (%f, %f, %f)\n", name, f3[0], f3[1], f3[2]); }
__device__ void Print(const char* name, const Float4& f4) { printf("%s = (%f, %f, %f, %f)\n", name, f4[0], f4[1], f4[2], f4[3]); }