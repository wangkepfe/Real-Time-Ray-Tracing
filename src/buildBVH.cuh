#pragma once

#include "cuda_runtime.h"
#include "linear_math.h"
#include "bvhNode.cuh"

__device__ __inline__ int LCP(uint* morton, uint triCount, int m0, int j)
{
	int res;
	if (j < 0 || j >= triCount) { res = 0; }
	else { res = __clz(m0 ^ morton[j]); }
	return res;
}

__global__ void BuildLBVH (BVHNode* bvhNodes, AABB* aabbs, uint* morton, uint* reorderIdx,/* uint* bvhNodeParent,*/ uint* isAabbDone, uint triCount)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= triCount - 1) { return; }

	//-------------------------------------------------------------------------------------------------------------------------
	// https://developer.nvidia.com/blog/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
	// Determine direction of the range (+1 or -1)
	int m0 = morton[i];
	int deltaLeft = LCP(morton, triCount, m0, i - 1);
	int deltaRight = LCP(morton, triCount, m0, i + 1);
	int d = (deltaRight - deltaLeft) >= 0 ? 1 : -1;

	// Compute upper bound for the length of the range
	int deltaMin = LCP(morton, triCount, m0, i - d);
	int lmax = 2;
	while (LCP(morton, triCount, m0, i + lmax * d) > deltaMin)
	{
		lmax *= 2;
	}

	// Find the other end using binary search
	int l = 0;
	for (int t = lmax / 2; t >= 1; t /= 2)
	{
		if (LCP(morton, triCount, m0, i + (l + t) * d) > deltaMin)
		{
			l += t;
		}
	}
	int j = i + l * d;

	// Find the split position using binary search
	int deltaNode = LCP(morton, triCount, m0, j);
	int s = 0;
	int div = 2;
	int t = (l + div - 1) / div;
	while (t >= 1)
	{
		if (LCP(morton, triCount, m0, i + (s + t) * d) > deltaNode)
		{
			s += t;
		}
		div *= 2;
		t = (l + div - 1) / div;
	}
	int gamma = i + s * d + min(d, 0);

	// Output child pointers. the children of Ii cover the ranges [min(i, j), γ] and [γ + 1,max(i, j)]
	if (min(i, j) == gamma)
	{
		bvhNodes[i].isLeftLeaf = 1;
		bvhNodes[i].idxLeft = reorderIdx[gamma];
	}
	else
	{
		bvhNodes[i].isLeftLeaf = 0;
		bvhNodes[i].idxLeft = gamma;
		//bvhNodeParent[gamma] = i;
	}

	if (max(i, j) == gamma + 1)
	{
		bvhNodes[i].isRightLeaf = 1;
		bvhNodes[i].idxRight = reorderIdx[gamma + 1];
	}
	else
	{
		bvhNodes[i].isRightLeaf = 0;
		bvhNodes[i].idxRight = gamma + 1;
		//bvhNodeParent[gamma + 1] = i;
	}
	//-------------------------------------------------------------------------------------------------------------------------

	while(atomicCAS(&isAabbDone[0], 0, 0) == 0)
	{
		if (((bvhNodes[i].isLeftLeaf == 1)  || (bvhNodes[i].isLeftLeaf == 0  && atomicCAS(&isAabbDone[bvhNodes[i].idxLeft], 0, 0) == 1)) &&
		    ((bvhNodes[i].isRightLeaf == 1) || (bvhNodes[i].isRightLeaf == 0 && atomicCAS(&isAabbDone[bvhNodes[i].idxRight], 0, 0) == 1)))
		{
			AABB leftAabb  = (bvhNodes[i].isLeftLeaf  == 1) ? aabbs[bvhNodes[i].idxLeft]  : bvhNodes[bvhNodes[i].idxLeft ].aabb.GetMerged();
			AABB rightAabb = (bvhNodes[i].isRightLeaf == 1) ? aabbs[bvhNodes[i].idxRight] : bvhNodes[bvhNodes[i].idxRight].aabb.GetMerged();

			bvhNodes[i].aabb = AABBCompact(leftAabb, rightAabb);

			atomicExch(&isAabbDone[i], 1);
		}
	}
}