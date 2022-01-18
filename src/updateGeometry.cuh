#pragma once

#include "cuda_runtime.h"
#include "linear_math.h"
#include "geometry.cuh"
#include <assert.h>
#include "precision.cuh"

// ------------------------------ Morton Code 3D ----------------------------------------------------
// 32bit 3D morton code encode
// --------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint MortonCode3D(uint x, uint y, uint z) {
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y << 8)) & 0x0300F00F;
	y = (y | (y << 4)) & 0x030C30C3;
	y = (y | (y << 2)) & 0x09249249;
	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z << 8)) & 0x0300F00F;
	z = (z | (z << 4)) & 0x030C30C3;
	z = (z | (z << 2)) & 0x09249249;
	return x | (y << 1) | (z << 2);
}

__inline__ __device__ void WarpReduceMaxMin3f(Float3& vmax, Float3& vmin) {
	const int warpSize = 32;
	#pragma unroll
	for (uint offset = warpSize / 2; offset > 0; offset /= 2)
	{
		vmin.x = min(__shfl_down_sync(0xffffffff, vmin.x, offset), vmin.x);
		vmin.y = min(__shfl_down_sync(0xffffffff, vmin.y, offset), vmin.y);
		vmin.z = min(__shfl_down_sync(0xffffffff, vmin.z, offset), vmin.z);

		vmax.x = max(__shfl_down_sync(0xffffffff, vmax.x, offset), vmax.x);
		vmax.y = max(__shfl_down_sync(0xffffffff, vmax.y, offset), vmax.y);
		vmax.z = max(__shfl_down_sync(0xffffffff, vmax.z, offset), vmax.z);
	}
}

template<uint kernelSize,          // thread number per kernel, same as sharedAabb size, sharedAabb-thread 1-to-1 mapping
         uint perThreadBatch>      // batch process count per thread
__global__ void UpdateSceneGeometry(
	Triangle*    constTriangles,   // [const] reference mesh
	Triangle*    triangles,        // [out] animated mesh
	AABB*        aabbs,            // [out] per triangle aabb
	uint*        morton,           // [out] per triangle motron code
	uint*        triCountArray,
	float        clockTime)        // for animation
{
	const bool applyTransform = false;

	const uint blocksize = kernelSize * perThreadBatch;

	uint objectId = blockIdx.x;
	uint triStart = objectId * blocksize;
	uint triCount = triCountArray[objectId];

	constTriangles   += triStart;
	triangles        += triStart;
	aabbs            += triStart;
	morton           += triStart;

	__shared__ AABB sceneBoundingBox[1];
	__shared__ AABB sharedAabb[kernelSize];

	sharedAabb[threadIdx.x] = AABB();

	if (threadIdx.x * perThreadBatch > triCount - 1)
		return;

	// index mapping, 1-to-many
	uint idx[perThreadBatch];
	#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		idx[i] = threadIdx.x * perThreadBatch + i;
	}

	// ------------------------------------ update triangle position ------------------------------------
	Triangle mytriangle[perThreadBatch];
	#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		mytriangle[i] = constTriangles[idx[i]];

		if (applyTransform)
		{
			Float3 v1 = mytriangle[i].v1;
			Float3 v2 = mytriangle[i].v2;
			Float3 v3 = mytriangle[i].v3;

			v1.y += 1.2f;
			v2.y += 1.2f;
			v3.y += 1.2f;

			Mat3 rotMat = RotationMatrixY(clockTime  * TWO_PI / 50.0);

			v1 = rotMat * v1;
			v2 = rotMat * v2;
			v3 = rotMat * v3;

			mytriangle[i] = Triangle(v1, v2, v3);
		}

	#if RAY_TRIANGLE_COORDINATE_TRANSFORM
		PreCalcTriangleCoordTrans(mytriangle[i]);
	#endif

		// write out
		triangles[idx[i]] = mytriangle[i];
	}

	// ------------------------------------ update aabb ------------------------------------
	AABB currentAABB[perThreadBatch];
	Float3 triangleCenter[perThreadBatch];
	#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		Float3 v1 = mytriangle[i].v1;
		Float3 v2 = mytriangle[i].v2;
		Float3 v3 = mytriangle[i].v3;

		Float3 aabbmin = min3f(v1, min3f(v2, v3));
		Float3 aabbmax = max3f(v1, max3f(v2, v3));

		// padding for precision issue
		Float3 diff = aabbmax - aabbmin;
		diff = max3f(Float3(0.001f), diff);
		aabbmax = aabbmin + diff + MachineEpsilon() * aabbmax;

		currentAABB[i] = AABB(aabbmax, aabbmin);
		triangleCenter[i] = (v1 + v2 + v3) / 3.0f;

		// write out
		aabbs[idx[i]] = currentAABB[i];
	}

	// ------------------------------------ per thread batch bounding box ------------------------------------
	AABB currentBatchAABB = currentAABB[0];
	#pragma unroll
	for (uint i = 1; i < perThreadBatch; ++i)
	{
		currentBatchAABB.min = min3f(currentBatchAABB.min, currentAABB[i].min);
		currentBatchAABB.max = max3f(currentBatchAABB.max, currentAABB[i].max);
	}

	// ------------------------------------ reduce across threads for scene bounding box ------------------------------------
	

	// thread id: tid
	uint tid = threadIdx.x;

	// init sharedAabb with AABB of per thread batch
	sharedAabb[tid] = currentBatchAABB;

	__syncthreads();

	// Reduce across warps
	#pragma unroll
	for (uint stride = kernelSize / 2; stride >= 64; stride >>= 1)
	{
		if (kernelSize > stride && tid < stride && tid + stride < triCount)
		{
			sharedAabb[tid].min = min3f(sharedAabb[tid].min, sharedAabb[tid + stride].min);
			sharedAabb[tid].max = max3f(sharedAabb[tid].max, sharedAabb[tid + stride].max);
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		sharedAabb[tid].min = min3f(sharedAabb[tid].min, sharedAabb[tid + 32].min);
		sharedAabb[tid].max = max3f(sharedAabb[tid].max, sharedAabb[tid + 32].max);
	}

	// Reduce inside warps
	AABB sceneAabb;

	if (tid < 32)
	{
		sceneAabb = sharedAabb[tid];
		WarpReduceMaxMin3f(sceneAabb.max, sceneAabb.min);
	}
	__syncthreads();

	// write out
	if (tid == 0)
	{
		sceneBoundingBox[0] = sceneAabb;
		#if DEBUG_BVH_BUILD
		Print("sceneAabb.min", sceneAabb.min);
		Print("sceneAabb.max", sceneAabb.max);
		#endif
	}
	__syncthreads();

	// broadcast to all threads
	sceneAabb = sceneBoundingBox[0];

	__syncthreads();

	// ------------------------------------ assign morton code to aabb ------------------------------------
	#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		Float3 unitBox = (triangleCenter[i] - sceneAabb.min) / (sceneAabb.max - sceneAabb.min);

		uint mortonCode = MortonCode3D((uint)(unitBox.x * 1023.0f),
								       (uint)(unitBox.y * 1023.0f),
								       (uint)(unitBox.z * 1023.0f));

		morton[idx[i]] = mortonCode;
	}
}

template<uint kernelSize,
         uint perThreadBatch,
		 uint batchSize>
__global__ void UpdateTLAS(
	BVHNode* bvhNodes,    // [in]
	AABB*    aabbs,       // [out]
	uint*    morton,      // [out]
	uint*    batchCount,  // [in]
	uint     tricountpadded)  // [in]
{
	uint tid = threadIdx.x;
	uint triCount = *batchCount;
	__shared__ AABB sceneBoundingBox[1];
	__shared__ AABB sharedAabb[kernelSize];
	sharedAabb[tid] = AABB();

	if (tid * perThreadBatch > triCount - 1)
		return;

	// get aabb of blas
	AABB currentAABB[perThreadBatch];
	Float3 blasCenter[perThreadBatch];
	AABB currentBatchAABB = AABB();
	#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		uint idx = batchSize * (tid * perThreadBatch + i);

		if (idx < tricountpadded)
		{
			BVHNode topBvhNodeInBlas = bvhNodes[idx];

			currentAABB[i] = topBvhNodeInBlas.aabb.GetMerged();
			blasCenter[i] = (currentAABB[i].max + currentAABB[i].min) / 2.0f;

			// write out
			aabbs[tid * perThreadBatch + i] = currentAABB[i];

			if (i == 0)
			{
				currentBatchAABB = currentAABB[0];
			}
			else
			{
				currentBatchAABB.min = min3f(currentBatchAABB.min, currentAABB[i].min);
				currentBatchAABB.max = max3f(currentBatchAABB.max, currentAABB[i].max);
			}
		}
	}

	// get scene aabb
	sharedAabb[tid] = currentBatchAABB;

	// Reduce across warps
	__syncthreads();
	#pragma unroll
	for (uint stride = kernelSize / 2; stride >= 64; stride >>= 1)
	{
		if (kernelSize > stride && tid < stride && tid + stride < triCount)
		{
			sharedAabb[tid].min = min3f(sharedAabb[tid].min, sharedAabb[tid + stride].min);
			sharedAabb[tid].max = max3f(sharedAabb[tid].max, sharedAabb[tid + stride].max);
		}
		__syncthreads();
	}

	// Reduce inside warps
	AABB sceneAabb;
	if (tid < 32)
	{
		sceneAabb = sharedAabb[tid];
		WarpReduceMaxMin3f(sceneAabb.max, sceneAabb.min);
	}
	__syncthreads();

	// broadcast to all threads
	if (tid == 0)
	{
		sceneBoundingBox[0] = sceneAabb;
	}
	__syncthreads();
	sceneAabb = sceneBoundingBox[0];

	// ------------------------------------ assign morton code to aabb ------------------------------------
	#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		uint idx = batchSize * (tid * perThreadBatch + i);

		if (idx < tricountpadded)
		{
			Float3 unitBox = (blasCenter[i] - sceneAabb.min) / (sceneAabb.max - sceneAabb.min);

			uint mortonCode = MortonCode3D((uint)(unitBox.x * 1023.0f),
										(uint)(unitBox.y * 1023.0f),
										(uint)(unitBox.z * 1023.0f));

			morton[tid * perThreadBatch + i] = mortonCode;
		}
	}
}
