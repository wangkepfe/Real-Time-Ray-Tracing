#pragma once

#include "cuda_runtime.h"
#include "linearMath.h"
#include "bvhNode.cuh"

// LCP Longest Common Prefix
__device__ __inline__ int LCP(uint* morton, uint triCount, int m0, int j)
{
	int res;
	if (j < 0 || j >= triCount) { res = 0; }
	else { res = __clz(m0 ^ morton[j]); }
	return res;
}

template<uint kernelSize,          // thread number per kernel, same as LDS size, LDS-thread 1-to-1 mapping
         uint perThreadBatch>      // batch process count per thread
__global__ void BuildLBVH (
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

	// For triCount == 1, TLAS is ok, BLAS is not ok: AABBCompact needs at least two triangles!!
	if (triCount == 1)
	{
		bvhNodes[0].aabb = AABBCompact(aabbs[0], AABB(0, 0));
		bvhNodes[0].idxLeft = 0;
		bvhNodes[0].idxRight = 0;
		bvhNodes[0].isLeftLeaf = 1;
		bvhNodes[0].isRightLeaf = 1;
	}

	bvhNodes   += triStart;
	aabbs      += triStart;
	morton     += triStart;
	reorderIdx += triStart;

	uint tid = threadIdx.x;

	if (tid * perThreadBatch > triCount - 1)
		return;

	__shared__ ushort parentIdx[kernelSize * perThreadBatch];
	__shared__ uint tempIdx[kernelSize * perThreadBatch];

	// Init lds!!
	for (uint k = 0; k < perThreadBatch; ++k)
	{
		uint i = tid * perThreadBatch + k;
		tempIdx[i] = 0;
	}

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

	// Find valid workitems
	bool isValid[perThreadBatch];
	for (uint j = 0; j < perThreadBatch; ++j)
	{
		uint i = tid * perThreadBatch + j;

		if (i >= triCount - 1)
		{
			// out of index
			isValid[j] = false;
		}
		else
		{
			// We start work with nodes with two leaves
			if (bvhNodes[i].isLeftLeaf && bvhNodes[i].isRightLeaf)
			{
				bvhNodes[i].aabb = AABBCompact(aabbs[bvhNodes[i].idxLeft], aabbs[bvhNodes[i].idxRight]);
				isValid[j] = true;
			}
			else
			{
				// Otherwise no work to do
				isValid[j] = false;
			}
		}
	}

	// early return for thread that doens't have any work to do
	uint countValid = 0;
	for (uint j = 0; j < perThreadBatch; ++j)
	{
		if (isValid[j])
		{
			countValid++;
		}
	}
	if (countValid == 0)
		return;

	// Set index for the workitems
	uint idx[perThreadBatch];
	for (uint j = 0; j < perThreadBatch; ++j)
	{
		idx[j] = tid * perThreadBatch + j;
	}

	// Keep merging, until top has a bvh
	bool reachedTopNode = false;
	while (reachedTopNode == false && countValid != 0)
	{
		// Reset valid workitem count
		countValid = 0;

		for (uint j = 0; j < perThreadBatch; ++j)
		{
			if (isValid[j] == false)
			{
				continue;
			}

			// Add to valie workitem count
			countValid++;

			// Save current node idx
			uint thisIndex = idx[j];

			// Go to parent node
			uint i = parentIdx[idx[j]];

			// Save parent node index
			idx[j] = i;


			// If the node has one leaf
			if (bvhNodes[i].isLeftLeaf)
			{
				bvhNodes[i].aabb = AABBCompact(aabbs[bvhNodes[i].idxLeft], bvhNodes[thisIndex].aabb.GetMerged());
				#if DEBUG_BVH_BUILD
				printf("tid = %d, thisIndex = %d, i = %d, left is leaf\n", tid, thisIndex, i);
				#endif
			}
			else if (bvhNodes[i].isRightLeaf)
			{
				bvhNodes[i].aabb = AABBCompact(bvhNodes[thisIndex].aabb.GetMerged(), aabbs[bvhNodes[i].idxRight]);
				#if DEBUG_BVH_BUILD
				printf("tid = %d, thisIndex = %d, i = %d, right is leaf\n", tid, thisIndex, i);
				#endif
			}
			// If the node has two children nodes
			else
			{
				// Two nodes will be here, the first one should record index and return, the second node can process
				uint theOtherIndex = atomicCAS(&tempIdx[i], 0, thisIndex);

				if (theOtherIndex == 0)
				{
					// The first node will read other index as zero
					isValid[j] = false;
					countValid--;
					#if DEBUG_BVH_BUILD
					printf("tid = %d, thisIndex = %d, i = %d, both node, first, return\n", tid, thisIndex, i);
					#endif
					continue;
				}
				else
				{
					// The second node processes BVH merge and continue in loop
					if (bvhNodes[i].idxLeft == thisIndex)
					{
						bvhNodes[i].aabb = AABBCompact(bvhNodes[thisIndex].aabb.GetMerged(), bvhNodes[theOtherIndex].aabb.GetMerged());
					}
					else
					{
						bvhNodes[i].aabb = AABBCompact(bvhNodes[theOtherIndex].aabb.GetMerged(), bvhNodes[thisIndex].aabb.GetMerged());
					}
					#if DEBUG_BVH_BUILD
					printf("tid = %d, thisIndex = %d, i = %d, both node, second, continue\n", tid, thisIndex, i);
					#endif
				}
			}

			// After processing BVH merge, check if we are at the top
			if (i == 0)
			{
				reachedTopNode = true;
			}
		}
	}
	#if DEBUG_BVH_BUILD
	printf("tid = %d, reachedTopNode = %d, countValid = %d\n", tid, reachedTopNode, countValid);
	#endif
}