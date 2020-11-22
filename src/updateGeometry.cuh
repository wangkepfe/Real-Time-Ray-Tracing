#pragma once

#include "cuda_runtime.h"
#include "linear_math.h"
#include "geometry.cuh"

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

template<uint lds_size>
__global__ void UpdateSceneGeometry(Triangle* constTriangles, Triangle* triangles, AABB* aabbs, AABB* sceneBoundingBox, uint* morton, unsigned int triCount, float clockTime)
{
	uint idx = threadIdx.x;
	if (idx > triCount - 1) return;

	// ------------------------------------ update triangle position ------------------------------------
#if 0
	// a plane
	uint edgeLen = (uint)sqrtf((float)triCount);
	uint idxx = idx / edgeLen;
	uint idxy = idx % edgeLen;
	Float3 startPos = {-0.05, 0, 0.001 };
	Float3 endPos1 = {-0.05, 0.1, 0.1 };
	Float3 endPos2 = {0.05, 0, 0.001 };
	Float3 lineStep1 = (endPos1 - startPos) / (float)edgeLen;
	Float3 lineStep2 = (endPos2 - startPos) / (float)edgeLen;
	Float3 v1 = startPos + lineStep1 * idxx + lineStep2 * idxy;

	Float3 v2Offset = {-0.005, 0, 0};
	Float3 v3Offset = {0, -0.005, 0};
	Float3 v2 = v1 + v2Offset;
	Float3 v3 = v1 + v3Offset;

	Triangle mytriangle(v1, v2, v3);
#endif

	Triangle mytriangle = constTriangles[idx];

	Float3 v1 = mytriangle.v1;
	Float3 v2 = mytriangle.v2;
	Float3 v3 = mytriangle.v3;

	Float3 n1 = mytriangle.n1;
	Float3 n2 = mytriangle.n2;
	Float3 n3 = mytriangle.n3;

	v1.y += 1.2f;
	v2.y += 1.2f;
	v3.y += 1.2f;

	Mat3 rotMat = RotationMatrixY(clockTime  * TWO_PI / 50.0);

	v1 = rotMat * v1;
	v2 = rotMat * v2;
	v3 = rotMat * v3;

	n1 = rotMat * n1;
	n2 = rotMat * n2;
	n3 = rotMat * n3;

	mytriangle = Triangle(v1, v2, v3);

#if RAY_TRIANGLE_COORDINATE_TRANSFORM
	PreCalcTriangleCoordTrans(mytriangle);
#endif

	mytriangle.n1 = n1;
	mytriangle.n2 = n2;
	mytriangle.n3 = n3;

	// write out
	triangles[idx] = mytriangle;

	// ------------------------------------ update aabb ------------------------------------
	Float3 aabbmin = min3f(v1, min3f(v2, v3));
	Float3 aabbmax = max3f(v1, max3f(v2, v3));

	Float3 diff = aabbmax - aabbmin;
	diff = max3f(Float3(0.001f), diff);
	aabbmax = aabbmin + diff;

	AABB currentAABB = AABB(aabbmax, aabbmin);
	aabbs[idx] = currentAABB;
	Float3 aabbcenter = (aabbmax + aabbmin) / 2.0f;

	// ------------------------------------ reduce for scene bounding box ------------------------------------
	AABB sceneAabb;

	__shared__ AABB lds[lds_size];
	lds[idx] = currentAABB;
	__syncthreads();

	#pragma unroll
	for (uint stride = 512; stride >= 64; stride >>= 1)
	{
		if (lds_size > stride && idx < stride && idx + stride < triCount)
		{
			lds[idx].min = min3f(lds[idx].min, lds[idx + stride].min);
			lds[idx].max = max3f(lds[idx].max, lds[idx + stride].max);
		}
		__syncthreads();
	}

	if (idx < 32)
	{
		lds[idx].min = min3f(lds[idx].min, lds[idx + 32].min);
		lds[idx].max = max3f(lds[idx].max, lds[idx + 32].max);

		WarpReduceMaxMin3f(lds[idx].max, lds[idx].min);
	}
	__syncthreads();

	sceneAabb = lds[0];

	if (idx == 0)
	{
		sceneBoundingBox[0] = lds[0];
	}
	__syncthreads();

	// ------------------------------------ assign morton code to aabb ------------------------------------
	Float3 unitBox = (aabbcenter - sceneAabb.min) / (sceneAabb.max - sceneAabb.min);

	uint mortonCode = MortonCode3D((uint)(unitBox.x * 1023.0f),
	                               (uint)(unitBox.y * 1023.0f),
								   (uint)(unitBox.z * 1023.0f));

	morton[idx] = mortonCode;
}