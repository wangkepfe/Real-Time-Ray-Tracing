#pragma once

#include "geometry.cuh"
#include "bvhNode.cuh"
#include "debugUtil.h"
#include "precision.cuh"

// BVH traverse stack
union BvhNodeStackNode
{
	struct
	{
		uint idx : 15;
		uint blasOffset : 15;
		uint isBlas : 1;
		uint isLeaf : 1;
		float t;
	};
	uint uint32All[2];
};

__host__ __device__ inline void Print(const char* name, const BvhNodeStackNode& node) {
	printf("%s = BvhNodeStackNode { idx=%d, blasOffset=%d, isBlas=%d, isLeaf=%d, t=%f }\n", name, node.idx, node.blasOffset, node.isBlas, node.isLeaf, node.t);
}

struct BvhNodeStack
{
	static const int bvhTraverseStackSize = 16;

	__host__ __device__ BvhNodeStackNode& operator[](int idx) { return data[idx]; }

	__host__ __device__ void push(BvhNodeStackNode node)
	{
		if (stackTop >= bvhTraverseStackSize - 1) {
			Print("BVH TRAVERSAL ERROR: push to full stack!!");
		} else {
			++stackTop;
			data[stackTop] = node;
		}
	}

	__host__ __device__ BvhNodeStackNode pop()
	{
		if (isEmpty()) {
			Print("BVH TRAVERSAL ERROR: pop empty stack!!");
			return {};
		} else {
			BvhNodeStackNode node = data[stackTop];
			--stackTop;
			return node;
		}
	}

	__host__ __device__ void push(int _idx, int _blasOffset, int _isBlas, int _isLeaf, float _t)
	{
		if (stackTop >= bvhTraverseStackSize - 1) {
			Print("BVH TRAVERSAL ERROR: push to full stack!!");
		} else {
			++stackTop;
			data[stackTop].idx        = _idx;
			data[stackTop].blasOffset = _blasOffset;
			data[stackTop].isBlas     = _isBlas;
			data[stackTop].isLeaf     = _isLeaf;
			data[stackTop].t          = _t;
		}
	}

	__host__ __device__ void pop(int& _idx, int& _blasOffset, int& _isBlas, int& _isLeaf, float& _t)
	{
		if (isEmpty()) {
			Print("BVH TRAVERSAL ERROR: pop empty stack!!");
		} else {
			_idx        = data[stackTop].idx;
			_blasOffset = data[stackTop].blasOffset;
			_isBlas     = data[stackTop].isBlas;
			_isLeaf     = data[stackTop].isLeaf;
			_t          = data[stackTop].t;
			--stackTop;
		}
	}

	__host__ __device__ bool isEmpty() { return stackTop < 0; }

	int stackTop = -1;
	BvhNodeStackNode data[bvhTraverseStackSize];
};

__device__ __host__ inline bool TestForFinish(BvhNodeStack& bvhNodeStack, BvhNodeStackNode& curr, float t)
{
	bool finished = false;
	do
	{
		// if stack is empty, finished
		if (bvhNodeStack.isEmpty())
		{
			finished = true;
			return finished;
		}

		// if not, go fetch one in stack
		curr = bvhNodeStack.pop();
	} while (curr.t > t); // if the one in stack is further than the closest hit, continue

	return finished;
}

__device__ __host__ inline void TraverseBvh(
	Triangle*     triangles,
	BVHNode*      bvhNodes,
	BVHNode*      tlasBvhNodes,
	uint          bvhBatchSize,
	const Ray&    ray,
	const Float3& invRayDir,
	float&        errorP,
	float&        errorT,
	float&        t,
	Float2&       uv,
	int&          objectIdx,
	Float3&       intersectNormal,
	Float3&       fakeNormal,
	Float3&       intersectPoint,
	float&        rayOffset,
	int           bvhNodesSize,
	int           tlasBvhNodesSize,
	int           trianglesSize)
{
	BVHNode topNode = tlasBvhNodes[0];
	AABB sceneBox = topNode.aabb.GetMerged();
	auto rayBoxIntersectionHelper = CreateRayBoxIntersectionHelper(ray, sceneBox, invRayDir);

	// BVH traversal
	const int maxBvhTraverseLoop = 1024;

	bool intersect1, intersect2, isClosestIntersect1;

	float t1;
	float t2;

	BvhNodeStack bvhNodeStack = {};

	BvhNodeStackNode curr = {};
	curr.idx = 0;
	curr.blasOffset = 0;
	curr.isBlas = 0;
	curr.isLeaf = 0;
	curr.t = -FLT_MAX;

	for (int i = 0; i < maxBvhTraverseLoop; ++i)
	{
		if (curr.isLeaf)
		{
			if (curr.isBlas)
			{
				int loadIdx = curr.blasOffset * bvhBatchSize + curr.idx;
				Triangle tri = SAFE_LOAD(triangles, loadIdx, trianglesSize, Triangle{});

				// triangle test
				float t_temp = RayTriangleIntersect(ray, tri, t, uv.x, uv.y, errorT);

				// hit
				if (t_temp < t)
				{
					t               = t_temp;
					objectIdx       = loadIdx;
					intersectNormal = cross(tri.v2 - tri.v1, tri.v3 - tri.v1).normalize();
					intersectPoint  = GetRayPlaneIntersectPoint(Float4(intersectNormal, -dot(intersectNormal, tri.v1)), ray, t, errorP);
					rayOffset       = errorT + errorP;

					#if USE_INTERPOLATED_FAKE_NORMAL
					fakeNormal = normalize(tri.n1 * (1.0f - uv.x - uv.y) + tri.n2 * uv.x + tri.n3 * uv.y);
					#endif

					#if RAY_TRIANGLE_COORDINATE_TRANSFORM
					intersectNormal = normalize(cross(tri.v2, tri.v3));
					intersectPoint  = GetRayPlaneIntersectPoint(Float4(intersectNormal, -dot(intersectNormal, tri.v4)), ray, t, errorP);
					#endif

					// DEBUG_PRINT(t);
					// DEBUG_PRINT(objectIdx);
					// DEBUG_PRINT(intersectNormal);
					// DEBUG_PRINT(intersectPoint);
					// DEBUG_PRINT(rayOffset);
				}

				if (TestForFinish(bvhNodeStack, curr, t))
					break;
			}
			else
			{
				curr.isLeaf = 0;
				curr.isBlas = 1;
				curr.blasOffset = curr.idx;
				curr.idx = 0;
			}
		}
		else
		{
			BVHNode currNode;

			if (curr.isBlas)
			{
				currNode = SAFE_LOAD(bvhNodes, curr.blasOffset * bvhBatchSize + curr.idx, bvhNodesSize, BVHNode{});
			}
			else
			{
				currNode = SAFE_LOAD(tlasBvhNodes, curr.idx, tlasBvhNodesSize, BVHNode{});
			}

			// test two aabb
			RayAabbPairIntersect(rayBoxIntersectionHelper, invRayDir, ray.orig, currNode.aabb, intersect1, intersect2, t1, t2);

			if (!intersect1 && !intersect2) // no hit for both sides
			{
				if (TestForFinish(bvhNodeStack, curr, t))
					break;
			}
			else if (intersect1 && !intersect2) // left hit only, go left
			{
				curr.idx = currNode.idxLeft;
				curr.isLeaf = currNode.isLeftLeaf;
				curr.t = t1;
			}
			else if (!intersect1 && intersect2) // right hit only, go right
			{
				curr.idx = currNode.idxRight;
				curr.isLeaf = currNode.isRightLeaf;
				curr.t = t2;
			}
			else // both hit
			{
				if (t1 < t2) // left is closer. go left next, push right to stack
				{
					bvhNodeStack.push(currNode.idxRight, curr.blasOffset, curr.isBlas, currNode.isRightLeaf, t2);

					curr.idx = currNode.idxLeft;
					curr.isLeaf = currNode.isLeftLeaf;
					curr.t = t1;
				}
				else // right is closer. go right next, push left to stack
				{
					bvhNodeStack.push(currNode.idxLeft, curr.blasOffset, curr.isBlas, currNode.isLeftLeaf, t1);

					curr.idx = currNode.idxRight;
					curr.isLeaf = currNode.isRightLeaf;
					curr.t = t2;
				}
			}
		}
	}
}