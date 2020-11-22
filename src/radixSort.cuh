#pragma once

#include "cuda_runtime.h"
#include "linear_math.h"

template<typename T>
__device__ __forceinline__ void ReduceAddToRightVec(volatile T* buffer, uint i, uint n, uint vecIdx, uint vecLen, uint& step, uint& stepBit, bool canReadWrite)
{
	stepBit = 1;
	step = 1;
	#pragma unroll
	while (step < n)
	{
		if (canReadWrite && i < (n >> stepBit))
		{
			uint idxR = n - 1 - 2 * i * step;
			uint idxL = idxR - step;

			idxR = idxR * vecLen + vecIdx;
			idxL = idxL * vecLen + vecIdx;

			buffer[idxR] += buffer[idxL];
		}
		__syncthreads();

		step <<= 1;
		++stepBit;
	}
	step >>= 1;
	--stepBit;
}

template<typename T>
__device__ __forceinline__ void ClearAndRecordSumVec(volatile T* buffer, volatile T* recordBuffer, uint i, uint n, uint vecIdx, uint vecLen, bool canReadWrite)
{
	if (canReadWrite && i == (n >> 1) - 1)
	{
		uint lastIdx = (n - 1) * vecLen + vecIdx;
		T sum = buffer[lastIdx];
		buffer[lastIdx] = 0;
		recordBuffer[vecIdx] = sum;
	}
	__syncthreads();
}

template<typename T>
__device__ __forceinline__ void ScanOnReducedVec(volatile T* buffer, uint i, uint n, uint vecIdx, uint vecLen, uint& step, uint& stepBit, bool canReadWrite)
{
	#pragma unroll
	while (step >= 1)
	{
		if (canReadWrite && i < (n >> stepBit))
		{
			uint idxR = n - 1 - 2 * i * step;
			uint idxL = idxR - step;

			idxR = idxR * vecLen + vecIdx;
			idxL = idxL * vecLen + vecIdx;

			T left = buffer[idxL];
			T right = buffer[idxR];

			buffer[idxL] = right;
			buffer[idxR] = left + right;
		}
		__syncthreads();
		step >>= 1;
		--stepBit;
	}
}

template<typename T>
__device__ __forceinline__ void InclusiveScan16(T& v, uint laneId)
{
	T v1 = __shfl_sync(0xffffffff, v, laneId - 1); v = v1;
	if (laneId == 0) { v = 0; }

	v1 = __shfl_sync(0xffffffff, v, laneId - 1); if (laneId > 0) { v += v1; }
	v1 = __shfl_sync(0xffffffff, v, laneId - 2); if (laneId > 1) { v += v1; }
	v1 = __shfl_sync(0xffffffff, v, laneId - 4); if (laneId > 3) { v += v1; }
	v1 = __shfl_sync(0xffffffff, v, laneId - 8); if (laneId > 7) { v += v1; }
}

template<uint lds_size>
__global__ void RadixSort(uint* inout, uint* reorderIdx)
{
	struct LDS
	{
		uint tempIdx[lds_size];
		uint temp[lds_size];
		ushort histo[16 * (lds_size / 32)]; // 16 values per warp
		ushort histoScan[16]; // 16 in total
	};

	volatile __shared__ LDS lds;
	lds.tempIdx[threadIdx.x] = threadIdx.x;

	//------------------------------------ Read data in ----------------------------------------
    lds.temp[threadIdx.x] = inout[threadIdx.x];
	__syncthreads();

	// lane id and warp id
	uint laneId = threadIdx.x & 0x1f;
	uint warpId = (threadIdx.x & 0xffffffe0) >> 5u;
	uint warpCount = lds_size >> 5u;

	// loop for each 4 bits
	#pragma unroll
	for (uint bitOffset = 0; bitOffset < 4 * 8; bitOffset += 4)
	{
		// load number
		uint num = lds.temp[threadIdx.x];

		// extract 4 bits
		uint num4bit = (num & (0xf << bitOffset)) >> bitOffset;

		//------------------------------------ Init LDS ----------------------------------------
		if (laneId < 16) { lds.histo[warpId * 16 + laneId] = 0; }
		if (warpId == 0 && laneId < 16) { lds.histoScan[laneId] = 0; }
		__syncthreads();

		//------------------------------------ Warp count and offset, by polling for same number to the current thread ----------------------------------------

		// mask indicates threads having equal value number with current thread
		uint mask = 0xffffffff;
		#pragma unroll
		for (int i = 0; i < 4; ++i)
		{
			uint bitPred = num4bit & (0x1 << i);
			uint maskOne = __ballot_sync(0xffffffff, bitPred);
			mask = mask & (bitPred ? maskOne : ~maskOne);
		}

		// offset of current value number
		uint pos = __popc(mask & (0xffffffff >> (31u - laneId)));

		// count of current value number
		uint count = __popc(mask);

		//------------------------------------ Scan across the warps, each has 16 values for number count within a warp ----------------------------------------

		if (pos == 1) { lds.histo[warpId * 16 + num4bit] = count; }
		__syncthreads();

		bool canReadWrite = (laneId < 16 && warpId < (warpCount / 2));

		uint stepBit, step;
		ReduceAddToRightVec(lds.histo, warpId, warpCount, laneId, 16, step, stepBit, canReadWrite);

		ClearAndRecordSumVec(lds.histo, lds.histoScan, warpId, warpCount, laneId, 16, canReadWrite);

		ScanOnReducedVec(lds.histo, warpId, warpCount, laneId, 16, step, stepBit, canReadWrite);

		//------------------------------------ Scan 16 values for total count of 16 numbers ----------------------------------------
		if (warpId == (warpCount >> 1) - 1)
		{
			uint v = (laneId < 16) ? lds.histoScan[laneId] : 0;
			InclusiveScan16(v, laneId);
			if (laneId < 16) { lds.histoScan[laneId] = v; }
		}
		__syncthreads();

		//------------------------------------ Reorder ----------------------------------------
		uint idxAllNum          = lds.histoScan[num4bit];
		uint idxCurrentNumBlock = lds.histo[warpId * 16 + num4bit];
		uint idxCurrentNumWarp  = pos - 1;

		uint finalIdx = idxAllNum + idxCurrentNumBlock + idxCurrentNumWarp;

		// write num
		lds.temp[finalIdx] = num;

		// read & write reorderIdx
		uint currentReorderIdx = lds.tempIdx[threadIdx.x];
		__syncthreads();
		lds.tempIdx[finalIdx] = currentReorderIdx;
		__syncthreads();
	}

	//------------------------------------ Write out ----------------------------------------
	inout[threadIdx.x] = lds.temp[threadIdx.x];
	reorderIdx[threadIdx.x] = lds.tempIdx[threadIdx.x];
}