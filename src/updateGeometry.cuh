#pragma once

#include "cuda_runtime.h"
#include "linear_math.h"
#include "geometry.cuh"
#include <assert.h>

// ------------------------------ Morton Code 3D ----------------------------------------------------
// 32bit 3D morton code encode
// --------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned int MortonCode3D(unsigned int x, unsigned int y, unsigned int z) {
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
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		vmin.x = min(__shfl_down_sync(0xffffffff, vmin.x, offset), vmin.x);
		vmin.y = min(__shfl_down_sync(0xffffffff, vmin.y, offset), vmin.y);
		vmin.z = min(__shfl_down_sync(0xffffffff, vmin.z, offset), vmin.z);

		vmax.x = max(__shfl_down_sync(0xffffffff, vmax.x, offset), vmax.x);
		vmax.y = max(__shfl_down_sync(0xffffffff, vmax.y, offset), vmax.y);
		vmax.z = max(__shfl_down_sync(0xffffffff, vmax.z, offset), vmax.z);
	}
}

template<uint kernelSize,          // thread number per kernel, same as LDS size, LDS-thread 1-to-1 mapping
         uint perThreadBatch>      // batch process count per thread
__global__ void UpdateSceneGeometry(
	Triangle*    constTriangles,   // [const] reference mesh
	Triangle*    triangles,        // [out] animated mesh
	AABB*        aabbs,            // [out] per triangle aabb
	AABB*        sceneBoundingBox, // [out] size = 1
	uint*        morton,           // [out] per triangle motron code
	uint         triCount,         // triangle count
	float        clockTime)        // for animation
{
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

		Float3 v1 = mytriangle[i].v1;
		Float3 v2 = mytriangle[i].v2;
		Float3 v3 = mytriangle[i].v3;

		Float3 n1 = mytriangle[i].n1;
		Float3 n2 = mytriangle[i].n2;
		Float3 n3 = mytriangle[i].n3;

		// v1.y += 1.2f;
		// v2.y += 1.2f;
		// v3.y += 1.2f;

		// Mat3 rotMat = RotationMatrixY(clockTime  * TWO_PI / 50.0);

		// v1 = rotMat * v1;
		// v2 = rotMat * v2;
		// v3 = rotMat * v3;

		// n1 = rotMat * n1;
		// n2 = rotMat * n2;
		// n3 = rotMat * n3;

		mytriangle[i] = Triangle(v1, v2, v3);

	#if RAY_TRIANGLE_COORDINATE_TRANSFORM
		PreCalcTriangleCoordTrans(mytriangle[i]);
	#endif

		mytriangle[i].n1 = n1;
		mytriangle[i].n2 = n2;
		mytriangle[i].n3 = n3;

		// write out
		triangles[idx[i]] = mytriangle[i];
	}

	// ------------------------------------ update aabb ------------------------------------
	AABB currentAABB[perThreadBatch];
	Float3 aabbcenter[perThreadBatch];
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
		aabbmax = aabbmin + diff;

		currentAABB[i] = AABB(aabbmax, aabbmin);
		aabbcenter[i] = (aabbmax + aabbmin) / 2.0f;

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
	__shared__ AABB lds[kernelSize];

	// thread id: tid
	uint tid = threadIdx.x;

	// init lds with AABB of per thread batch
	lds[tid] = currentBatchAABB;

	__syncthreads();

	// Reduce across warps
	#pragma unroll
	for (uint stride = kernelSize / 2; stride >= 64; stride >>= 1)
	{
		if (kernelSize > stride && tid < stride && tid + stride < triCount)
		{
			lds[tid].min = min3f(lds[tid].min, lds[tid + stride].min);
			lds[tid].max = max3f(lds[tid].max, lds[tid + stride].max);
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		lds[tid].min = min3f(lds[tid].min, lds[tid + 32].min);
		lds[tid].max = max3f(lds[tid].max, lds[tid + 32].max);
	}

	// Reduce inside warps
	AABB sceneAabb;

	if (tid < 32)
	{
		sceneAabb = lds[tid];
		WarpReduceMaxMin3f(sceneAabb.max, sceneAabb.min);
	}
	__syncthreads();

	// write out
	if (tid == 0)
	{
		sceneBoundingBox[0] = sceneAabb;
	}
	__syncthreads();

	// broadcast to all threads
	sceneAabb = sceneBoundingBox[0];

	__syncthreads();

	// ------------------------------------ assign morton code to aabb ------------------------------------
	#pragma unroll
	for (uint i = 0; i < perThreadBatch; ++i)
	{
		Float3 unitBox = (aabbcenter[i] - sceneAabb.min) / (sceneAabb.max - sceneAabb.min);

		uint mortonCode = MortonCode3D((uint)(unitBox.x * 1023.0f),
								       (uint)(unitBox.y * 1023.0f),
								       (uint)(unitBox.z * 1023.0f));

		morton[idx[i]] = mortonCode;
	}
}
