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

__device__ __host__ inline const float TwoPowerMinus23()
{
	int_32 v {0x34000000};
	return v.f32;
}

__device__ __host__ inline const float TwoPowerMinus24()
{
	int_32 v {0x33800000};
	return v.f32;
}

__device__ __host__ inline const float p()
{
	return 1.0f + TwoPowerMinus23();
}

__device__ __host__ inline const float m()
{
	return 1.0f - TwoPowerMinus23();
}

__device__ __host__ inline float up(float a)
{
	return a>0.0f ? a*p() : a*m();
}

__device__ __host__ inline float dn(float a)
{
	return a>0.0f ? a*m() : a*p();
}

__device__ __host__ inline float Up(float a) { return a*p(); }
__device__ __host__ inline float Dn(float a) { return a*m(); }

__device__ __host__ inline Float3 Up(Float3 a) { return Float3(Up(a.x), Up(a.y), Up(a.z)); }
__device__ __host__ inline Float3 Dn(Float3 a) { return Float3(Dn(a.x), Dn(a.y), Dn(a.z)); }

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
	int kz = max_dim(abs(ray.dir));
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

	if (ray.dir[kz] < 0.0f) swap(kx,ky);

	Int3 nearID(0,1,2);
	Int3 farID(3,4,5);
	int nearX = nearID[kx], farX = farID[kx];
	int nearY = nearID[ky], farY = farID[ky];
	int nearZ = nearID[kz], farZ = farID[kz];
	if (ray.dir[kx] < 0.0f) swap(nearX,farX);
	if (ray.dir[ky] < 0.0f) swap(nearY,farY);
	if (ray.dir[kz] < 0.0f) swap(nearZ,farZ);

	BVHNode topNode = tlasBvhNodes[0];
	AABB sceneBox = topNode.aabb.GetMerged();

	const float eps = 5.0f * TwoPowerMinus24();
	Float3 lower = Dn(abs(ray.orig-sceneBox.min));
	Float3 upper = Up(abs(ray.orig-sceneBox.max));
	float max_z = max(lower[kz],upper[kz]);

	float err_near_x = Up(lower[kx]+max_z);
	float err_near_y = Up(lower[ky]+max_z);
	float org_near_x = up(ray.orig[kx]+Up(eps*err_near_x));
	float org_near_y = up(ray.orig[ky]+Up(eps*err_near_y));
	float org_near_z = ray.orig[kz];
	float err_far_x = Up(upper[kx]+max_z);
	float err_far_y = Up(upper[ky]+max_z);
	float org_far_x = dn(ray.orig[kx]-Up(eps*err_far_x));
	float org_far_y = dn(ray.orig[ky]-Up(eps*err_far_y));
	float org_far_z = ray.orig[kz];

	if (ray.dir[kx] < 0.0f) swap(org_near_x,org_far_x);
	if (ray.dir[ky] < 0.0f) swap(org_near_y,org_far_y);

	float rdir_near_x = Dn(Dn(invRayDir[kx]));
	float rdir_near_y = Dn(Dn(invRayDir[ky]));
	float rdir_near_z = Dn(Dn(invRayDir[kz]));
	float rdir_far_x = Up(Up(invRayDir[kx]));
	float rdir_far_y = Up(Up(invRayDir[ky]));
	float rdir_far_z = Up(Up(invRayDir[kz]));

	RayBoxIntersectionHelper rayBoxIntersectionHelper;
	rayBoxIntersectionHelper.farX = farX;
	rayBoxIntersectionHelper.farY = farY;
	rayBoxIntersectionHelper.farZ = farZ;
	rayBoxIntersectionHelper.nearX = nearX;
	rayBoxIntersectionHelper.nearY = nearY;
	rayBoxIntersectionHelper.nearZ = nearZ;
	rayBoxIntersectionHelper.org_near_x = org_near_x;
	rayBoxIntersectionHelper.org_near_y = org_near_y;
	rayBoxIntersectionHelper.org_near_z = org_near_z;
	rayBoxIntersectionHelper.org_far_x = org_far_x;
	rayBoxIntersectionHelper.org_far_y = org_far_y;
	rayBoxIntersectionHelper.org_far_z = org_far_z;
    rayBoxIntersectionHelper.rdir_near_x = rdir_near_x;
    rayBoxIntersectionHelper.rdir_near_y = rdir_near_y;
    rayBoxIntersectionHelper.rdir_near_z = rdir_near_z;
    rayBoxIntersectionHelper.rdir_far_x  = rdir_far_x ;
    rayBoxIntersectionHelper.rdir_far_y  = rdir_far_y ;
    rayBoxIntersectionHelper.rdir_far_z  = rdir_far_z ;

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