#pragma once

#include "cuda_runtime.h"
#include "linearMath.h"

template<typename T, uint size>
__device__ __forceinline__ void SimdInclusiveScan(T& v, uint laneId)
{
	T v1 = __shfl_sync(0xffffffff, v, laneId - 1); v = v1;
	if (laneId == 0) { v = 0; }

	if (size > 1) { v1 = __shfl_sync(0xffffffff, v, laneId - 1); if (laneId > 0) { v += v1; } }
	if (size > 2) { v1 = __shfl_sync(0xffffffff, v, laneId - 2); if (laneId > 1) { v += v1; } }
	if (size > 4) { v1 = __shfl_sync(0xffffffff, v, laneId - 4); if (laneId > 3) { v += v1; } }
	if (size > 8) { v1 = __shfl_sync(0xffffffff, v, laneId - 8); if (laneId > 7) { v += v1; } }
	if (size > 16) { v1 = __shfl_sync(0xffffffff, v, laneId - 16); if (laneId > 15) { v += v1; } }
}

template<uint kernelSize,          // thread number per kernel, same as LDS size, LDS-thread 1-to-1 mapping
	uint perThreadBatch>      // batch process count per thread
	__global__ void RadixSort(
		uint* inout,
		uint* reorderIdx)
{
	const uint blocksize = kernelSize * perThreadBatch;
	uint objectId = blockIdx.x;
	uint triStart = objectId * blocksize;;
	inout += triStart;
	reorderIdx += triStart;

	// warp count
	const uint warpCount = kernelSize >> 5;

	// thread id
	uint tid = threadIdx.x;

	// lane id and warp id
	uint laneId = tid & 0x1f;
	uint warpId = tid >> 5;

	// lds
	struct LDS
	{
		// temporary value holder
		uint   temp[perThreadBatch * kernelSize];

		// temporary reorderIdx holder
		ushort tempIdx[perThreadBatch * kernelSize];

		// 4-bit 16 values per warp
		ushort histogramScan[perThreadBatch * warpCount + 1][32];
	};
	volatile __shared__ LDS lds;

	//------------------------------------ Read data in ----------------------------------------

#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		uint id = warpId * 32 * perThreadBatch + i * 32 + laneId;

		lds.tempIdx[id] = id;
		lds.temp[id] = inout[id];
	}

	__syncthreads();

	//------------------------------------ loop for each 4 bits ----------------------------------------
#pragma unroll
	for (uint bitOffset = 0; bitOffset < 4 * 8; bitOffset += 4)
	{
		//------------------------------------ Init LDS ----------------------------------------
#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			uint wid = warpId * perThreadBatch + i;

			lds.histogramScan[wid + 1][laneId] = 0;
		}

		if (warpId == 0)
		{
			lds.histogramScan[0][laneId] = 0;
		}

		//------------------------------------ load number ----------------------------------------
		uint num[perThreadBatch];
		uint num4bit[perThreadBatch];

#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			uint id = warpId * 32 * perThreadBatch + i * 32 + laneId;

			// copy global to lds
			num[i] = lds.temp[id];

			// extract 4 bits
			num4bit[i] = (num[i] & (0xf << bitOffset)) >> bitOffset;
		}
		__syncthreads();

		//------------------------------------ Warp count and offset, by polling for same number to the current thread ----------------------------------------
		uint pos[perThreadBatch];
		uint count[perThreadBatch];

#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			// mask indicates threads having equal value number with current thread
			uint mask = 0xffffffff;
#pragma unroll
			for (uint bit = 0; bit < 4; ++bit)
			{
				uint bitPred = num4bit[i] & (0x1 << bit);
				uint maskOne = __ballot_sync(0xffffffff, bitPred);
				mask = mask & (bitPred ? maskOne : ~maskOne);
			}

			// index in an array formed by the same value in the group
			// value 0 1 2 3 2 2 1 2
			// pos   1 1 1 1 2 3 2 4
			pos[i] = __popc(mask & (0xffffffff >> (31u - laneId)));

			// total count of the same value number in the group
			// value 0 1 2 3 2 2 1 2
			// count 1 2 4 1 4 4 2 4
			count[i] = __popc(mask);
		}

		//------------------------------------ Write simd local result to lds ------------------------------------

#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			uint wid = warpId * perThreadBatch + i;

			// Each warp has 32 numbers, the numbers are ranged in 16 values, each value has a count of numbers
			if (pos[i] == 1)
			{
				lds.histogramScan[wid + 1][num4bit[i]] = count[i];
			}
		}
		__syncthreads();

		//------------------------------------ Scan across the warps, each has 16 values for number count within a warp ----------------------------------------
		uint tempBlockSum[perThreadBatch];

#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			uint wid = warpId * perThreadBatch + i;

			tempBlockSum[i] = lds.histogramScan[wid + 1][laneId];
		}

#pragma unroll
		for (uint offset = 1; offset < warpCount * perThreadBatch; offset *= 2)
		{
			__syncthreads();

#pragma unroll
			for (uint i = 0; i < perThreadBatch; ++i)
			{
				uint wid = warpId * perThreadBatch + i;

				if (wid >= offset)
				{
					tempBlockSum[i] += lds.histogramScan[wid - offset + 1][laneId];
				}
			}

			__syncthreads();

#pragma unroll
			for (uint i = 0; i < perThreadBatch; ++i)
			{
				uint wid = warpId * perThreadBatch + i;

				lds.histogramScan[wid + 1][laneId] = tempBlockSum[i];
			}
		}

		__syncthreads();

		//------------------------------------ Scan 16 values for total count of 16 numbers ----------------------------------------
		if (warpId == warpCount - 1)
		{
			uint v = lds.histogramScan[warpCount * perThreadBatch][laneId];

			SimdInclusiveScan<uint, 16>(v, laneId);

			lds.histogramScan[warpCount * perThreadBatch][laneId] = v;
		}
		__syncthreads();

		//------------------------------------ Reorder ----------------------------------------
		uint finalIdx[perThreadBatch];

#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			uint wid = warpId * perThreadBatch + i;

			uint idxAllNum = lds.histogramScan[warpCount * perThreadBatch][num4bit[i]];
			uint idxCurrentNumBlock = lds.histogramScan[wid][num4bit[i]];
			uint idxCurrentNumWarp = pos[i] - 1;

			finalIdx[i] = idxAllNum + idxCurrentNumBlock + idxCurrentNumWarp;

			// write num
			lds.temp[finalIdx[i]] = num[i];
		}

		// read reorderIdx
		uint currentReorderIdx[perThreadBatch];

#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			uint id = warpId * 32 * perThreadBatch + i * 32 + laneId;
			currentReorderIdx[i] = lds.tempIdx[id];
		}

		__syncthreads();

		// write reorderIdx
#pragma unroll
		for (uint i = 0; i < perThreadBatch; ++i)
		{
			lds.tempIdx[finalIdx[i]] = currentReorderIdx[i];
		}

		__syncthreads();
	}

	//------------------------------------ Write out ----------------------------------------
#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		uint id = warpId * 32 * perThreadBatch + i * 32 + laneId;

		inout[id] = lds.temp[id];
		reorderIdx[id] = lds.tempIdx[id];
	}
}
