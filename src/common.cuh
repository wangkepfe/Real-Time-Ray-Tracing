#pragma once

#include <cuda_runtime.h>

template<int n>
__global__ void PrefixScan(float *data)
{
	// allocated on invocation
	extern __shared__ float temp[];

	int i = threadIdx.x;
	int offset = 1;

	// load input into shared memory
	temp[2 * i] = data[2 * i];
	temp[2 * i + 1] = data[2 * i + 1];

	// save orig value
	float orig0 = temp[2 * i];
	float orig1 = temp[2 * i + 1];

	// build sum in place up the tree
	#pragma unroll
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (i < d)
		{
			int ai = offset * (2 * i + 1) - 1;
			int bi = offset * (2 * i + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// clear the last element
	if (i == 0)
	{
		temp[n - 1] = 0;
	}

	// traverse down tree & build scan
	#pragma unroll
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (i < d)
		{
			int ai = offset * (2 * i + 1) - 1;
			int bi = offset * (2 * i + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	// write results to device memory
	data[2 * i] = temp[2 * i] + orig0;
	data[2 * i + 1] = temp[2 * i + 1] + orig1;

	//printf("data[%d] = %f\n", 2 * i, data[2 * i]);
	//("data[%d] = %f\n", 2 * i + 1, data[2 * i + 1]);
}

__device__ __forceinline__ float GetLuma(const Float3& rgb)
{
	return rgb.y * 2.0f + rgb.x + rgb.z;
}
