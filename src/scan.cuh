#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include "cudaError.cuh"

//----------------------------------------------------------------------------------------------------------------------------------------------
// Resolve LDS bank conflict by padding
#define NUM_BANKS 32 
#define LOG_NUM_BANKS 5
//#define CONFLICT_FREE_OFFSET(n) 0
#define CONFLICT_FREE_OFFSET(n) ((n) >> (LOG_NUM_BANKS * 2))
//#define CONFLICT_FREE_OFFSET(n) ((n) >> (NUM_BANKS) + (n) >> (LOG_NUM_BANKS * 2))

//----------------------------------------------------------------------------------------------------------------------------------------------
// Debug flag
#define DEBUG_SCAN_KERNEL 0

//----------------------------------------------------------------------------------------------------------------------------------------------
// Batch sizes and kernel sizes
// Can be tuned for better performance
namespace 
{
constexpr int supportedSize[13] =  { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
constexpr int batchPerThread[13] = { 1, 1, 1, 1,  1,  1,  1,   2,   4,   8,    8,    8,    8    };
constexpr int kernalSizes[13] =    { 1, 2, 4, 8,  16, 32, 64,  64,  64,  64,   128,  256,  512  };
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// Scan Single Block cuda kernel
template<int n, int batch, int postFix>
__global__ void ScanSingleBlock(float *data, float* out, float* tmp, bool outputSum)
{
	// allocated on invocation
	extern __shared__ float temp[];

	// index
	int tid = threadIdx.x;

	// offsetting buffers
	data += blockIdx.x * n;
	out += blockIdx.x * n;

	// load input into shared memory
	#pragma unroll
	for (int k = 0; k < batch; ++k)
	{
		int i = tid * batch + k;
		temp[2 * i + CONFLICT_FREE_OFFSET(2 * i)] = data[2 * i];
		temp[2 * i + 1 + CONFLICT_FREE_OFFSET(2 * i + 1)] = data[2 * i + 1];
	}

	// save orig value
	float orig[batch * 2];
	if (postFix)
	{
		#pragma unroll
		for (int k = 0; k < batch; ++k)
		{
			int i = tid * batch + k;
			orig[2 * i] = temp[2 * i + CONFLICT_FREE_OFFSET(2 * i)];
			orig[2 * i + 1] = temp[2 * i + 1 + CONFLICT_FREE_OFFSET(2 * i + 1)];
		}
	}	

	// build sum in place up the tree
	int offset = 1;
	#pragma unroll
	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < batch; ++k)
		{
			int i = tid * batch + k;
			if (i < d)
			{
				int ai = offset * (2 * i + 1) - 1;
				int bi = offset * (2 * i + 2) - 1;
				temp[bi + CONFLICT_FREE_OFFSET(bi)] += temp[ai + CONFLICT_FREE_OFFSET(ai)];
			}
		}
		offset *= 2;
	}
	__syncthreads();

	// clear the last element
	if (tid == blockDim.x - 1)
	{
		if (outputSum)
		{
			tmp[blockIdx.x] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		}
		
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	// traverse down tree & build scan
	#pragma unroll
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < batch; ++k)
		{
			int i = tid * batch + k;
			if (i < d)
			{
				int ai = offset * (2 * i + 1) - 1;
				int bi = offset * (2 * i + 2) - 1;

				float t = temp[ai + CONFLICT_FREE_OFFSET(ai)];
				temp[ai + CONFLICT_FREE_OFFSET(ai)] = temp[bi + CONFLICT_FREE_OFFSET(bi)];
				temp[bi + CONFLICT_FREE_OFFSET(bi)] += t;
			}
		}
	}
	__syncthreads();

	// write results to device memory
	#pragma unroll
	for (int k = 0; k < batch; ++k)
	{
		int i = tid * batch + k;
		if (postFix)
		{
			out[2 * i] = temp[2 * i + CONFLICT_FREE_OFFSET(2 * i)] + orig[2 * i];
			out[2 * i + 1] = temp[2 * i + 1 + CONFLICT_FREE_OFFSET(2 * i + 1)] + orig[2 * i + 1];
		}
		else
		{
			out[2 * i] = temp[2 * i + CONFLICT_FREE_OFFSET(2 * i)];
			out[2 * i + 1] = temp[2 * i + 1 + CONFLICT_FREE_OFFSET(2 * i + 1)];
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// Array Sum cuda kernel
template<int batch>
__global__ void ScanPhaseArraySum(float *data, float* tmp)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	#pragma unroll
	for (int k = 0; k < batch; ++k)
	{
		int i = tid * batch + k;
		data[i * 2] += tmp[blockIdx.x];
		data[i * 2 + 1] += tmp[blockIdx.x];
	}
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// Scan kernel
inline void LaunchScanKernel(float* in, float* out, float* tmp, int blockSize, int size, bool outputSum, int postfix)
{
	int blockCount = size / blockSize;

	// Pad lds size for bank conflict resolve
	int paddedLdsSize[13] =  { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	for (int i = 0; i < 13; ++i)
	{
		paddedLdsSize[i] += CONFLICT_FREE_OFFSET(paddedLdsSize[i]);
	}

	if (postfix)
	{
		// Need to switch case these functions, since kernel size and batch size are template value, compiler will unroll the loop in the kernel
		switch (blockSize)
		{ 
		case 2:    ScanSingleBlock <supportedSize[0],  batchPerThread[0] , 1> <<<blockCount, kernalSizes[0],  paddedLdsSize[0]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 4:    ScanSingleBlock <supportedSize[1],  batchPerThread[1] , 1> <<<blockCount, kernalSizes[1],  paddedLdsSize[1]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 8:    ScanSingleBlock <supportedSize[2],  batchPerThread[2] , 1> <<<blockCount, kernalSizes[2],  paddedLdsSize[2]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 16:   ScanSingleBlock <supportedSize[3],  batchPerThread[3] , 1> <<<blockCount, kernalSizes[3],  paddedLdsSize[3]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 32:   ScanSingleBlock <supportedSize[4],  batchPerThread[4] , 1> <<<blockCount, kernalSizes[4],  paddedLdsSize[4]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 64:   ScanSingleBlock <supportedSize[5],  batchPerThread[5] , 1> <<<blockCount, kernalSizes[5],  paddedLdsSize[5]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 128:  ScanSingleBlock <supportedSize[6],  batchPerThread[6] , 1> <<<blockCount, kernalSizes[6],  paddedLdsSize[6]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 256:  ScanSingleBlock <supportedSize[7],  batchPerThread[7] , 1> <<<blockCount, kernalSizes[7],  paddedLdsSize[7]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 512:  ScanSingleBlock <supportedSize[8],  batchPerThread[8] , 1> <<<blockCount, kernalSizes[8],  paddedLdsSize[8]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 1024: ScanSingleBlock <supportedSize[9],  batchPerThread[9] , 1> <<<blockCount, kernalSizes[9],  paddedLdsSize[9]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 2048: ScanSingleBlock <supportedSize[10], batchPerThread[10], 1> <<<blockCount, kernalSizes[10], paddedLdsSize[10] * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 4096: ScanSingleBlock <supportedSize[11], batchPerThread[11], 1> <<<blockCount, kernalSizes[11], paddedLdsSize[11] * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 8192: ScanSingleBlock <supportedSize[12], batchPerThread[12], 1> <<<blockCount, kernalSizes[12], paddedLdsSize[12] * sizeof(float)>>> (in, out, tmp, outputSum); break;
		}
	}
	else
	{
		switch (blockSize)
		{ 
		case 2:    ScanSingleBlock <supportedSize[0],  batchPerThread[0] , 0> <<<blockCount, kernalSizes[0],  paddedLdsSize[0]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 4:    ScanSingleBlock <supportedSize[1],  batchPerThread[1] , 0> <<<blockCount, kernalSizes[1],  paddedLdsSize[1]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 8:    ScanSingleBlock <supportedSize[2],  batchPerThread[2] , 0> <<<blockCount, kernalSizes[2],  paddedLdsSize[2]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 16:   ScanSingleBlock <supportedSize[3],  batchPerThread[3] , 0> <<<blockCount, kernalSizes[3],  paddedLdsSize[3]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 32:   ScanSingleBlock <supportedSize[4],  batchPerThread[4] , 0> <<<blockCount, kernalSizes[4],  paddedLdsSize[4]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 64:   ScanSingleBlock <supportedSize[5],  batchPerThread[5] , 0> <<<blockCount, kernalSizes[5],  paddedLdsSize[5]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 128:  ScanSingleBlock <supportedSize[6],  batchPerThread[6] , 0> <<<blockCount, kernalSizes[6],  paddedLdsSize[6]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 256:  ScanSingleBlock <supportedSize[7],  batchPerThread[7] , 0> <<<blockCount, kernalSizes[7],  paddedLdsSize[7]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 512:  ScanSingleBlock <supportedSize[8],  batchPerThread[8] , 0> <<<blockCount, kernalSizes[8],  paddedLdsSize[8]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 1024: ScanSingleBlock <supportedSize[9],  batchPerThread[9] , 0> <<<blockCount, kernalSizes[9],  paddedLdsSize[9]  * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 2048: ScanSingleBlock <supportedSize[10], batchPerThread[10], 0> <<<blockCount, kernalSizes[10], paddedLdsSize[10] * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 4096: ScanSingleBlock <supportedSize[11], batchPerThread[11], 0> <<<blockCount, kernalSizes[11], paddedLdsSize[11] * sizeof(float)>>> (in, out, tmp, outputSum); break;
		case 8192: ScanSingleBlock <supportedSize[12], batchPerThread[12], 0> <<<blockCount, kernalSizes[12], paddedLdsSize[12] * sizeof(float)>>> (in, out, tmp, outputSum); break;
		}	
	}
} 

//----------------------------------------------------------------------------------------------------------------------------------------------
// Sum kernel
inline void LaunchSumKernel(float* out, float* tmp, int blockSize, int size)
{
	int blockCount = size / blockSize;

	// Need to switch case these functions, since kernel size and batch size are template value, compiler will unroll the loop in the kernel
	switch (blockSize)
	{
	case 2:    ScanPhaseArraySum <batchPerThread[0]>  <<<blockCount, kernalSizes[0],  supportedSize[0]  * sizeof(float)>>> (out, tmp); break;
	case 4:    ScanPhaseArraySum <batchPerThread[1]>  <<<blockCount, kernalSizes[1],  supportedSize[1]  * sizeof(float)>>> (out, tmp); break;
	case 8:    ScanPhaseArraySum <batchPerThread[2]>  <<<blockCount, kernalSizes[2],  supportedSize[2]  * sizeof(float)>>> (out, tmp); break;
	case 16:   ScanPhaseArraySum <batchPerThread[3]>  <<<blockCount, kernalSizes[3],  supportedSize[3]  * sizeof(float)>>> (out, tmp); break;
	case 32:   ScanPhaseArraySum <batchPerThread[4]>  <<<blockCount, kernalSizes[4],  supportedSize[4]  * sizeof(float)>>> (out, tmp); break;
	case 64:   ScanPhaseArraySum <batchPerThread[5]>  <<<blockCount, kernalSizes[5],  supportedSize[5]  * sizeof(float)>>> (out, tmp); break;
	case 128:  ScanPhaseArraySum <batchPerThread[6]>  <<<blockCount, kernalSizes[6],  supportedSize[6]  * sizeof(float)>>> (out, tmp); break;
	case 256:  ScanPhaseArraySum <batchPerThread[7]>  <<<blockCount, kernalSizes[7],  supportedSize[7]  * sizeof(float)>>> (out, tmp); break;
	case 512:  ScanPhaseArraySum <batchPerThread[8]>  <<<blockCount, kernalSizes[8],  supportedSize[8]  * sizeof(float)>>> (out, tmp); break;
	case 1024: ScanPhaseArraySum <batchPerThread[9]>  <<<blockCount, kernalSizes[9],  supportedSize[9]  * sizeof(float)>>> (out, tmp); break;
	case 2048: ScanPhaseArraySum <batchPerThread[10]> <<<blockCount, kernalSizes[10], supportedSize[10] * sizeof(float)>>> (out, tmp); break;
	case 4096: ScanPhaseArraySum <batchPerThread[11]> <<<blockCount, kernalSizes[11], supportedSize[11] * sizeof(float)>>> (out, tmp); break;
	case 8192: ScanPhaseArraySum <batchPerThread[12]> <<<blockCount, kernalSizes[12], supportedSize[12] * sizeof(float)>>> (out, tmp); break;
	}	
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// CPU prefix/postfix scan/sum
inline void CpuScan(float* in, float* out, int size, int postfix)
{
    float accu = 0;
    for (int i = 0; i < size; ++i)
    {
        if (postfix)
        {
            accu += in[i];
            out[i] = accu;
        }
        else
        {
            out[i] = accu;
            accu += in[i];
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// GPU prefix/postfix scan/sum
// Scan for each block, and scan for sum of each block, finally add them together
// Each scan needs a size of power of two! So the input size must be [power of two] x [power of two]
// "tmp" is the sum of each block, the size should be equal to blockCount=size/blockSize
inline void Scan(float* in, float* out, float* tmp, int size, int blockSize, int postfix)
{
	const int maxSupportedSize = 8192 * 8192;
	assert(size <= maxSupportedSize);
	assert((blockSize & (blockSize - 1)) == 0);

	int blockCount = size / blockSize;
	assert((blockCount & (blockCount - 1)) == 0);

	if (blockCount == 1)
	{
		LaunchScanKernel(in, out, nullptr, blockSize, size, 0, postfix);
	}
	else
	{
		// Scan for each block
		LaunchScanKernel(in, out, tmp, blockSize, size, 1, postfix);
		
		#if DEBUG_SCAN_KERNEL
		GpuErrorCheck(cudaDeviceSynchronize());
	    GpuErrorCheck(cudaPeekAtLastError());
		GpuErrorCheck(cudaMemcpy(h_out, tmp, blockCount * sizeof(float), cudaMemcpyDeviceToHost));
		PrintArray(h_out, blockCount, 1);
		CpuScan(h_out, cpuOut, blockCount, 0);
		PrintArray(cpuOut, blockCount, 1);
		#endif

		// Scan for the sum of each block, stored in "tmp"
		LaunchScanKernel(tmp, tmp, nullptr, blockCount, blockCount, 0, 0);

		#if DEBUG_SCAN_KERNEL
		GpuErrorCheck(cudaDeviceSynchronize());
	    GpuErrorCheck(cudaPeekAtLastError());
		GpuErrorCheck(cudaMemcpy(h_out, tmp, blockCount * sizeof(float), cudaMemcpyDeviceToHost));
		PrintArray(h_out, blockCount, 1);
		#endif

		// Add the value of scanned "tmp" to each blocks
		LaunchSumKernel(out, tmp, blockSize, size);
	}
}