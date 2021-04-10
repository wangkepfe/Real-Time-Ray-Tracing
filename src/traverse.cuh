#pragma once

#include "kernel.cuh"
#include "geometry.cuh"
#include "bvhNode.cuh"

#define DEBUG_BVH_TRAVERSE 0

// update material info after intersect
__device__ inline void UpdateMaterial(
	const ConstBuffer&   cbo,
	RayState&            rayState,
	const SceneMaterial& sceneMaterial,
	const SceneGeometry& sceneGeometry)
{
	// get mat id
	if (rayState.objectIdx == PLANE_OBJECT_IDX)
	{
		rayState.matId = 6;
	}
	else
	{
		rayState.matId = sceneMaterial.materialsIdx[rayState.objectIdx];
	}

	// get mat
	SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];

	// mat type
	rayState.matType = (rayState.hit == false) ? MAT_SKY : mat.type;

	// hit light
	if (rayState.matType == EMISSIVE)
	{
		if (rayState.lightIdx == rayState.objectIdx || rayState.lightIdx == DEFAULT_LIGHT_ID)
		{
			rayState.hitLight = true;
		}
		else
		{
			rayState.hitLight = false;
			rayState.matType = LAMBERTIAN_DIFFUSE;
		}
	}
	else if (rayState.matType == MAT_SKY)
	{
		rayState.hitLight = true;
	}

	// is diffuse
	rayState.isDiffuse = (rayState.matType == LAMBERTIAN_DIFFUSE) || (rayState.matType == MICROFACET_REFLECTION);
}

#define RAY_TRAVERSE_SPHERES 1
#define RAY_TRAVERSE_AABBS 0
#define RAY_TRAVERSE_PLANE 1
#define RAY_TRAVERSE_BVH 1

// ray traverse, return true if hit
__device__ inline void RaySceneIntersect(
	const ConstBuffer&   cbo,
	const SceneMaterial& sceneMaterial,
	const SceneGeometry& sceneGeometry,
	RayState&            rayState)
{
	if (rayState.hitLight == true) { return; }

	Ray ray(rayState.orig, rayState.dir);

	Float3& intersectPoint   = rayState.pos;
	Float3& intersectNormal  = rayState.normal;
	Float3& fakeNormal       = rayState.fakeNormal;
	int&    objectIdx        = rayState.objectIdx;
	float&  rayOffset        = rayState.offset;
	bool&   isRayIntoSurface = rayState.isRayIntoSurface;
	float&  normalDotRayDir  = rayState.normalDotRayDir;
	Float2& uv               = rayState.uv;

	// init error
	rayOffset    = 1e-7f;

	Float3 invRayDir = SafeDivide3f(Float3(1.0f), ray.dir);

	// init t with max distance
	float t         = RayMax;

	// init ray state
	objectIdx       = -1;
	intersectNormal = Float3(0);
	intersectPoint  = Float3(RayMax);
	uv              = Float2(0);

	// error
	float errorP = 1e-7f;
	float errorT = 1e-7f;

#if RAY_TRAVERSE_BVH
	Triangle* triangles = sceneGeometry.triangles;
	BVHNode* bvhNodes = sceneGeometry.bvhNodes;
	BVHNode* tlasBvhNodes = sceneGeometry.tlasBvhNodes;

	// BVH traversal
	static const int maxBvhTraverseLoop = 1024;
	static const int stackSize = 16;

	bool intersect1, intersect2, isClosestIntersect1;

	// stack
	union BvhNodeStackNode
	{
		struct
		{
			uint idx : 15;
			uint blasOffset : 15;
			uint isBlas : 1;
			uint isLeaf : 1;
		};
		uint uint32All;
	};

	struct BvhNodeStack
	{
		#if DEBUG_BVH_TRAVERSE
		__device__ BvhNodeStack()
		{
			stackTop = -1;
			for (int i = 0; i < stackSize; ++i)
			{
				data[i].uint32All = 0;
			}
		}
		#endif

		__device__ BvhNodeStackNode& operator[](int idx) { return data[idx]; }

		__device__ void push(BvhNodeStackNode node)
		{
			++stackTop;
			data[stackTop] = node;
		}

		__device__ BvhNodeStackNode pop()
		{
			BvhNodeStackNode node = data[stackTop];
			--stackTop;
			return node;
		}

		__device__ void push(int _idx, int _blasOffset, int _isBlas, int _isLeaf)
		{
			++stackTop;
			data[stackTop].idx = _idx;
			data[stackTop].blasOffset = _blasOffset;
			data[stackTop].isBlas = _isBlas;
			data[stackTop].isLeaf = _isLeaf;
		}

		__device__ void pop(int& _idx, int& _blasOffset, int& _isBlas, int& _isLeaf)
		{
			_idx = data[stackTop].idx;
			_blasOffset = data[stackTop].blasOffset;
			_isBlas = data[stackTop].isBlas;
			_isLeaf = data[stackTop].isLeaf;
			--stackTop;
		}

		__device__ bool isEmpty() { return stackTop < 0; }

		int stackTop = -1;
		BvhNodeStackNode data[stackSize];
	};

	BvhNodeStack bvhNodeStack;

	BvhNodeStackNode curr;
	curr.idx = 0;
	curr.blasOffset = 0;
	curr.isBlas = 0;
	curr.isLeaf = 0;

	for (int i = 0; i < maxBvhTraverseLoop; ++i)
	{
		#if DEBUG_BVH_TRAVERSE
		DEBUG_PRINT(i);
		DEBUG_PRINT(bvhNodeStack.stackTop);
		DEBUG_PRINT(curr.idx);
		DEBUG_PRINT(curr.blasOffset);
		DEBUG_PRINT(curr.isLeaf);
		DEBUG_PRINT(curr.isBlas);
		if (IS_DEBUG_PIXEL())
		{
			printf("bvhNodeStack.idx[] =    {%d, %d, %d, %d, ...}\n", bvhNodeStack[0].idx, bvhNodeStack[1].idx, bvhNodeStack[2].idx, bvhNodeStack[3].idx);
			printf("bvhNodeStack.isBlas[] = {%d, %d, %d, %d, ...}\n", bvhNodeStack[0].isBlas, bvhNodeStack[1].isBlas, bvhNodeStack[2].isBlas, bvhNodeStack[3].isBlas);
			printf("bvhNodeStack.isLeaf[] = {%d, %d, %d, %d, ...}\n", bvhNodeStack[0].isLeaf, bvhNodeStack[1].isLeaf, bvhNodeStack[2].isLeaf, bvhNodeStack[3].isLeaf);
		}
		#endif

		if (curr.isLeaf)
		{
			if (curr.isBlas)
			{
				int loadIdx = curr.blasOffset * cbo.bvhBatchSize + curr.idx;
				Triangle tri = triangles[loadIdx];

				// triangle test
				float t_temp = RayTriangleIntersect(ray, tri, t, uv.x, uv.y, errorT);

				// hit
				if (t_temp < t)
				{
					t               = t_temp;
					objectIdx       = loadIdx;

					#if RAY_TRIANGLE_MOLLER_TRUMBORE
					intersectNormal = cross(tri.v2 - tri.v1, tri.v3 - tri.v1).normalize();
					intersectPoint = GetRayPlaneIntersectPoint(Float4(intersectNormal, -dot(intersectNormal, tri.v1)), ray, t, errorP);
					#endif

					#if RAY_TRIANGLE_COORDINATE_TRANSFORM
					intersectNormal = normalize(cross(tri.v2, tri.v3));
					intersectPoint  = GetRayPlaneIntersectPoint(Float4(intersectNormal, -dot(intersectNormal, tri.v4)), ray, t, errorP);
					#endif

					#if USE_INTERPOLATED_FAKE_NORMAL
					fakeNormal = normalize(tri.n1 * (1.0f - uv.x - uv.y) + tri.n2 * uv.x + tri.n3 * uv.y);
					fakeNormal = dot(intersectNormal, fakeNormal) < 0 ? intersectNormal : fakeNormal;
					#endif

					rayOffset       = errorT + errorP;

					#if DEBUG_BVH_TRAVERSE
					DEBUG_PRINT(objectIdx);
					DEBUG_PRINT(tri.v1);
					DEBUG_PRINT(tri.v2);
					DEBUG_PRINT(tri.v3);
					DEBUG_PRINT(ray.orig);
					DEBUG_PRINT(ray.dir);
					DEBUG_PRINT(intersectNormal);
					DEBUG_PRINT(intersectPoint);
					DEBUG_PRINT(t);
					DEBUG_PRINT(rayOffset);
					DEBUG_PRINT_BAR
					#endif
				}
				else
				{
					#if DEBUG_BVH_TRAVERSE
					DEBUG_PRINT_STRING("Tested a triangle but no intersection!");
					DEBUG_PRINT(loadIdx)
					DEBUG_PRINT(t_temp)
					DEBUG_PRINT(t)
					DEBUG_PRINT_BAR
					#endif
				}

				// if stack is empty, finished
				if (bvhNodeStack.isEmpty())
				{
					#if DEBUG_BVH_TRAVERSE
					DEBUG_PRINT_STRING("Stack empty, quit");
					DEBUG_PRINT(objectIdx);
					DEBUG_PRINT_BAR
					#endif

					break;
				}

				// if not, go fetch one in stack
				curr = bvhNodeStack.pop();
			}
			else
			{
				curr.isLeaf = 0;
				curr.isBlas = 1;
				curr.blasOffset = curr.idx;
				curr.idx = 0;

				#if DEBUG_BVH_TRAVERSE
				DEBUG_PRINT_STRING("Go into BLAS");
				DEBUG_PRINT_BAR
				#endif
			}
		}
		else
		{
			BVHNode currNode;
			if (curr.isBlas)
			{
				int loadIdx = curr.blasOffset * cbo.bvhBatchSize + curr.idx;
				currNode = bvhNodes[loadIdx];

				#if DEBUG_BVH_TRAVERSE
				DEBUG_PRINT_STRING("Load BVH node from BLAS");
				DEBUG_PRINT(loadIdx);
				#endif
			}
			else
			{
				currNode = tlasBvhNodes[curr.idx];

				#if DEBUG_BVH_TRAVERSE
				DEBUG_PRINT_STRING("Load BVH node from TLAS");
				DEBUG_PRINT(curr.idx);
				#endif
			}

			// test two aabb
			RayAabbPairIntersect(invRayDir, ray.orig, currNode.aabb, intersect1, intersect2, isClosestIntersect1);

			#if DEBUG_BVH_TRAVERSE
			DEBUG_PRINT(currNode.idxLeft);
			DEBUG_PRINT(currNode.idxRight);
			DEBUG_PRINT(currNode.isLeftLeaf);
			DEBUG_PRINT(currNode.isRightLeaf);
			#endif

			if (!intersect1 && !intersect2) // no hit for both sides
			{
				// if stack empty, quit
				if (bvhNodeStack.isEmpty())
				{
					#if DEBUG_BVH_TRAVERSE
					DEBUG_PRINT_STRING("Stack empty, quit");
					DEBUG_PRINT_BAR
					#endif

					break;
				}

				// if not, get one from stack
				curr = bvhNodeStack.pop();
			}
			else if (intersect1 && !intersect2) // left hit
			{
				curr.idx = currNode.idxLeft;
				curr.isLeaf = currNode.isLeftLeaf;

				#if DEBUG_BVH_TRAVERSE
				if (i == cbo.bvhDebugLevel)
				{
					AABB testAabb = currNode.aabb.GetLeftAABB();
					float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
					if (t_temp < t)
					{
						t               = t_temp;
						objectIdx       = curr.idx;
						GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
						rayOffset       = errorT + errorP;
						DEBUG_PRINT(t);
					}
					break;
				}
				#endif
			}
			else if (!intersect1 && intersect2) // right hit
			{
				curr.idx = currNode.idxRight;
				curr.isLeaf = currNode.isRightLeaf;

				#if DEBUG_BVH_TRAVERSE
				if (i == cbo.bvhDebugLevel)
				{
					AABB testAabb = currNode.aabb.GetRightAABB();
					float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
					if (t_temp < t)
					{
						t               = t_temp;
						objectIdx       = curr.idx;
						GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
						rayOffset       = errorT + errorP;
						DEBUG_PRINT(t);
					}
					break;
				}
				#endif
			}
			else // both hit
			{
				int idx1, idx2;
				bool isLeaf1, isLeaf2;

				if (isClosestIntersect1)
				{
					// left is closer. go left next, push right to stack
					idx2    = currNode.idxRight;
					isLeaf2 = currNode.isRightLeaf;
					idx1    = currNode.idxLeft;
					isLeaf1 = currNode.isLeftLeaf;

					#if DEBUG_BVH_TRAVERSE
					if (i == cbo.bvhDebugLevel)
					{
						AABB testAabb = currNode.aabb.GetLeftAABB();
						float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
						if (t_temp < t)
						{
							t               = t_temp;
							objectIdx       = curr.idx;
							GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
							rayOffset       = errorT + errorP;
							DEBUG_PRINT(t);
						}
						break;
					}
					#endif
				}
				else
				{
					// right is closer. go right next, push left to stack
					idx2    = currNode.idxLeft;
					isLeaf2 = currNode.isLeftLeaf;
					idx1    = currNode.idxRight;
					isLeaf1 = currNode.isRightLeaf;

					#if DEBUG_BVH_TRAVERSE
					if (i == cbo.bvhDebugLevel)
					{
						AABB testAabb = currNode.aabb.GetRightAABB();
						float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
						if (t_temp < t)
						{
							t               = t_temp;
							objectIdx       = curr.idx;
							GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
							rayOffset       = errorT + errorP;
							DEBUG_PRINT(t);
						}
						break;
					}
					#endif
				}

				// push
				bvhNodeStack.push(idx2, curr.blasOffset, curr.isBlas, isLeaf2);

				curr.idx = idx1;
				curr.isLeaf = isLeaf1;

				#if DEBUG_BVH_TRAVERSE
				DEBUG_PRINT(bvhNodeStack.stackTop);
				DEBUG_PRINT(idx2);
				DEBUG_PRINT(idx1);
				#endif
			}
		}
	}
#endif

#if RAY_TRAVERSE_SPHERES
	// ----------------------- spheres ---------------------------
	// get spheres
	int numSpheres  = sceneGeometry.numSpheres;
	Sphere* spheres = sceneGeometry.spheres;

	// traverse sphere
	for (int i = 0; i < numSpheres; ++i)
	{
		const Sphere& sphere = spheres[i];

		// intersect sphere/ray, get distance t_temp
		float t_temp = SphereRayIntersect(sphere, ray, errorT);

		// update nearest hit
		if (t_temp < t)
		{
			t               = t_temp;
			objectIdx       = sceneGeometry.numTriangles + i;
			intersectPoint  = GetSphereRayIntersectPoint(sphere, ray, t, errorP);
			intersectNormal = normalize(intersectPoint - sphere.center);
			rayOffset       = errorT + errorP;
		}
	}
#endif

#if RAY_TRAVERSE_AABBS
	// ----------------------- aabbs ---------------------------
	// traverse aabb
	int numAabbs = sceneGeometry.numAabbs;
	AABB* aabbs = sceneGeometry.aabbs;

	// pre-calculation
	Float3 oneOverRayDir = invRayDir;
	Float3 rayOrigOverRayDir = ray.orig * oneOverRayDir;

	for (int i = 0; i < numAabbs; ++i)
	{
		AABB& aabb = aabbs[i];
		float t_temp = AabbRayIntersect(aabb.min, aabb.max, ray.orig, oneOverRayDir, rayOrigOverRayDir, errorT);
		if (t_temp < t)
		{
			t               = t_temp;
			objectIdx       = numSpheres + i;
			GetAabbRayIntersectPointNormal(aabb, ray, t, intersectPoint, intersectNormal, errorP);
			//GetAabbRayIntersectPointNormalUv(aabb, ray, t, intersectPoint, intersectNormal, uv, errorP);
			rayOffset       = errorT + errorP;
		}
	}
#endif

#if RAY_TRAVERSE_PLANE
	// ----------------------- planes ---------------------------
	Float4 plane(Float3(0, 1, 0), 0);
	float t_temp = RayPlaneIntersect(plane, ray, errorT);
	if (t_temp < t && t_temp < 1e3f)
	{
		t               = t_temp;
		objectIdx       = PLANE_OBJECT_IDX;
		intersectPoint  = GetRayPlaneIntersectPoint(plane, ray, t, errorP);
		intersectNormal = plane.xyz;
		rayOffset       = errorT + errorP;
	}

#endif

	// Is ray into surface? If no, flip normal
	normalDotRayDir = dot(intersectNormal, ray.dir);    // ray dot geometry normal
	isRayIntoSurface = normalDotRayDir < 0;     // ray shoot into surface, if dot < 0
	if (isRayIntoSurface == false)
	{
		intersectNormal = -intersectNormal; // if ray not shoot into surface, convert the case to "ray shoot into surface" by flip the normal
		normalDotRayDir = -normalDotRayDir;
	}

	rayState.hit = (t < RayMax);
	rayState.depth = t;

	UpdateMaterial(cbo, rayState, sceneMaterial, sceneGeometry);
}

