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

template<uint kernelSize,          // thread number per kernel, same as LDS size, LDS-thread 1-to-1 mapping
         uint perThreadBatch>      // batch process count per thread
__global__ void BuildLBVH2 (
	BVHNode* bvhNodes, // [out]
	AABB*    aabbs,       // [in]
	uint*    morton,      // [in]
	uint*    reorderIdx,  // [in]
	uint*    triCountArray)
{
	const uint blocksize = kernelSize * perThreadBatch;
	uint objectId = blockIdx.x;
	uint triStart = objectId * blocksize;
	uint triCount = triCountArray[objectId];

	bvhNodes   += triStart;
	aabbs      += triStart;
	morton     += triStart;
	reorderIdx += triStart;

	uint tid = threadIdx.x;

	if (tid * perThreadBatch > triCount - 1)
		return;

	__shared__ ushort parentIdx[kernelSize * perThreadBatch];
	__shared__ uint tempIdx[kernelSize * perThreadBatch];

	for (uint k = 0; k < perThreadBatch; ++k)
	{
		uint i = tid * perThreadBatch + k;

		if (i >= triCount - 1)
			break;

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

		tempIdx[i] = 0;

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
			parentIdx[gamma] = i;
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
			parentIdx[gamma + 1] = i;
		}
	}

	__syncthreads();

	//------------------------------------------------- bvh build ----------------------------------------------------------
	bool isValid[perThreadBatch];

	for (uint j = 0; j < perThreadBatch; ++j)
	{
		uint i = tid * perThreadBatch + j;

		if (i >= triCount - 1)
		{
			isValid[j] = false;
		}

		// The node with two leaves
		if (bvhNodes[i].isLeftLeaf && bvhNodes[i].isRightLeaf)
		{
			bvhNodes[i].aabb = AABBCompact(aabbs[bvhNodes[i].idxLeft], aabbs[bvhNodes[i].idxRight]);
			isValid[j] = true;
		}
		else
		{
			isValid[j] = false;
		}
	}

	// keep merging, until top has a bvh
	bool finished = false;
	uint countValid = 0;
	uint idx[perThreadBatch];

	for (uint j = 0; j < perThreadBatch; ++j)
	{
		idx[j] = tid * perThreadBatch + j;

		if (isValid[j])
		{
			countValid++;
		}
	}
	finished = (countValid == 0);

	while(!finished)
	{
		countValid = 0;

		for (uint j = 0; j < perThreadBatch; ++j)
		{
			if (isValid[j])
			{
				countValid++;

				uint i = idx[j];

				// save current node idx
				uint thisIndex = i;

				// go to parent node
				i = parentIdx[i];

				idx[j] = i;

				// If the node has a leaf
				if (bvhNodes[i].isLeftLeaf)
				{
					bvhNodes[i].aabb = AABBCompact(aabbs[bvhNodes[i].idxLeft], bvhNodes[thisIndex].aabb.GetMerged());

				}
				else if (bvhNodes[i].isRightLeaf)
				{
					bvhNodes[i].aabb = AABBCompact(bvhNodes[thisIndex].aabb.GetMerged(), aabbs[bvhNodes[i].idxRight]);
				}
				// If the node has two children nodes
				else
				{
					// Two nodes will be here, the first one should record index and return, the second node can process
					uint theOtherIndex = atomicCAS(&tempIdx[i], 0, thisIndex);

					if (theOtherIndex == 0)
					{
						isValid[j] = false;
						countValid--;
					}
					else
					{
						if (bvhNodes[i].idxLeft == thisIndex)
						{
							bvhNodes[i].aabb = AABBCompact(bvhNodes[thisIndex].aabb.GetMerged(), bvhNodes[theOtherIndex].aabb.GetMerged());
						}
						else
						{
							bvhNodes[i].aabb = AABBCompact(bvhNodes[theOtherIndex].aabb.GetMerged(), bvhNodes[thisIndex].aabb.GetMerged());
						}

						if (i == 0)
						{
							finished = true;
						}
					}
				}
			}
		}

		if (!finished) finished = (countValid == 0);
	}
}