#pragma once

#include "kernel.cuh"
#include "geometry.cuh"
#include "bvhNode.cuh"

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

	// BVH traversal
	const int maxBvhTraverseLoop = 128;
	const int stackSize = 16;
	bool intersect1, intersect2, isClosestIntersect1;
	// init stack
	struct BvhNodeStack
	{
		__forceinline__ __device__ BvhNodeStack() : i32{0} {}

		__forceinline__ __device__ int   GetIdx()           { return i32 & 0x0000ffff; }
		__forceinline__ __device__ void  SetIdx(int i)      { i32 = (i32 & 0xffff0000) | (i & 0x0000ffff); }

		__forceinline__ __device__ int   GetIsLeaf()        { return (i32 >> 16); }
		__forceinline__ __device__ void  SetIsLeaf(int i)   { i32 = (i32 & 0x0000ffff) | (i << 16); }

		int i32;
	};
	BvhNodeStack bvhNodeStack[stackSize];
	// for (int i = 0; i < stackSize; ++i)
	// {
	// 	bvhNodeStack[i] = -1;
	// 	isLeaf[i] = -1;
	// }
	int stackTop = -1;
	int currIdx = 0;
	int isCurrLeaf = 0;
	for (int i = 0; i < maxBvhTraverseLoop; ++i)
	{

		// DEBUG_PRINT(i);
		// DEBUG_PRINT(stackTop);
		// DEBUG_PRINT(currIdx);
		// DEBUG_PRINT(isCurrLeaf);


		// if (IS_DEBUG_PIXEL())
		// {
		// 	printf("bvhNodeStack.idx[] =    {%d, %d, %d, %d, ...}\n", bvhNodeStack[0].GetIdx(), bvhNodeStack[1].GetIdx(), bvhNodeStack[2].GetIdx(), bvhNodeStack[3].GetIdx());
		// 	printf("bvhNodeStack.isLeaf[] = {%d, %d, %d, %d, ...}\n", bvhNodeStack[0].GetIsLeaf(), bvhNodeStack[1].GetIsLeaf(), bvhNodeStack[2].GetIsLeaf(), bvhNodeStack[3].GetIsLeaf());
		// 	bvhNodeStack[0].SetIsLeaf(1);
		// 	printf("bvhNodeStack.isLeaf[0] = %d\n", bvhNodeStack[0].GetIsLeaf());
		// }



		if (isCurrLeaf)
		{
			// Triangle tri;
			// if (currIdx < sceneGeometry.numTriangles && currIdx >= 0)
			// {
			// 	tri = triangles[currIdx];
			// }
			// else
			// {
			// 	Print("idx_error_tri", Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y));
			// 	break;
			// }
			Triangle tri = triangles[currIdx];

			// triangle test
			float t_temp = RayTriangleIntersect(ray, tri, t, uv.x, uv.y, errorT);

			// hit
			if (t_temp < t)
			{
				t               = t_temp;
				objectIdx       = currIdx;

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

				//DEBUG_PRINT(objectIdx);

				// DEBUG_PRINT(tri.v1);
				// DEBUG_PRINT(tri.v2);
				// DEBUG_PRINT(tri.v3);

				// DEBUG_PRINT(ray.orig);
				// DEBUG_PRINT(ray.dir);

				// DEBUG_PRINT(intersectNormal);
				// DEBUG_PRINT(intersectPoint);
				// DEBUG_PRINT(t);
				// DEBUG_PRINT(rayOffset);


				//DEBUG_PRINT_BAR

			    //break;
			}

			// pop
			if (stackTop < 0)
			{
				// if (IS_DEBUG_PIXEL())
				// {
				// DEBUG_PRINT(objectIdx);
				// }
				break;
			}

			currIdx = bvhNodeStack[stackTop].GetIdx();
			isCurrLeaf = bvhNodeStack[stackTop].GetIsLeaf();
			--stackTop;

			//currIdx = (stackTop >= 0 && stackTop < stackSize) ? bvhNodeStack[stackTop] : -1;
			//isCurrLeaf = (stackTop >= 0 && stackTop < stackSize) ? isLeaf[stackTop] : 0;
			// if (!(stackTop >= 0 && stackTop < stackSize))
			// {
			// 	Print("idx_error_stack1", Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y));
			// 	break;
			// }


		}
		else
		{
			// BVHNode currNode;
			// if (currIdx < sceneGeometry.numTriangles - 1 && currIdx >= 0)
			// {
			// 	currNode = bvhNodes[currIdx];
			// }
			// else
			// {
			// 	Print("idx_error_bvh", Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y));
			// 	break;
			// }
			BVHNode currNode = bvhNodes[currIdx];

			// test two aabb
			RayAabbPairIntersect(invRayDir, ray.orig, currNode.aabb, intersect1, intersect2, isClosestIntersect1);


				// DEBUG_PRINT(currNode.idxLeft);
				// DEBUG_PRINT(currNode.idxRight);
				// DEBUG_PRINT(currNode.isLeftLeaf);
				// DEBUG_PRINT(currNode.isRightLeaf);


			if (!intersect1 && !intersect2)
			{
				// no hit, pop
				if (stackTop < 0) { break; }

				currIdx = bvhNodeStack[stackTop].GetIdx();
				isCurrLeaf = bvhNodeStack[stackTop].GetIsLeaf();
				--stackTop;

				//currIdx = (stackTop >= 0 && stackTop < stackSize) ? bvhNodeStack[stackTop] : -1;
				//isCurrLeaf = (stackTop >= 0 && stackTop < stackSize) ? isLeaf[stackTop] : 0;
				// if (!(stackTop >= 0 && stackTop < stackSize))
				// {
				// 	Print("idx_error_stack2", Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y));
				// 	break;
				// }


			}
			else if (intersect1 && !intersect2)
			{
				// left hit
				currIdx = currNode.idxLeft;
				isCurrLeaf = currNode.isLeftLeaf;

				if (i == cbo.bvhDebugLevel)
				{
					AABB testAabb = currNode.aabb.GetLeftAABB();
					float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
					if (t_temp < t)
					{
						t               = t_temp;
						objectIdx       = currIdx;
						GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
						rayOffset       = errorT + errorP;
						//DEBUG_PRINT(t);
					}
					break;
				}
			}
			else if (!intersect1 && intersect2)
			{
				// right hit
				currIdx = currNode.idxRight;
				isCurrLeaf = currNode.isRightLeaf;

				if (i == cbo.bvhDebugLevel)
				{
					AABB testAabb = currNode.aabb.GetRightAABB();
					float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
					if (t_temp < t)
					{
						t               = t_temp;
						objectIdx       = currIdx;
						GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
						rayOffset       = errorT + errorP;
						//DEBUG_PRINT(t);
					}
					break;
				}
			}
			else
			{
				// both hit
				int idx1, idx2;
				bool isLeaf1, isLeaf2;

				if (isClosestIntersect1)
				{
					// left closer. push right
					idx2    = currNode.idxRight;
					isLeaf2 = currNode.isRightLeaf;
					idx1    = currNode.idxLeft;
					isLeaf1 = currNode.isLeftLeaf;

					// if (i == cbo.bvhDebugLevel)
					// {
					// 	AABB testAabb = currNode.aabb.GetLeftAABB();
					// 	float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
					// 	if (t_temp < t)
					// 	{
					// 		t               = t_temp;
					// 		objectIdx       = currIdx;
					// 		GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
					// 		rayOffset       = errorT + errorP;
					// 		DEBUG_PRINT(t);
					// 	}
					// 	break;
					// }
				}
				else
				{
					// right closer. push left
					idx2    = currNode.idxLeft;
					isLeaf2 = currNode.isLeftLeaf;
					idx1    = currNode.idxRight;
					isLeaf1 = currNode.isRightLeaf;

					// if (i == cbo.bvhDebugLevel)
					// {
					// 	AABB testAabb = currNode.aabb.GetRightAABB();
					// 	float t_temp = AabbRayIntersect(testAabb.min, testAabb.max, ray.orig, invRayDir, invRayDir * ray.orig, errorT);
					// 	if (t_temp < t)
					// 	{
					// 		t               = t_temp;
					// 		objectIdx       = currIdx;
					// 		GetAabbRayIntersectPointNormal(testAabb, ray, t, intersectPoint, intersectNormal, errorP);
					// 		rayOffset       = errorT + errorP;
					// 		DEBUG_PRINT(t);
					// 	}
					// 	break;
					// }
				}

				// push
				++stackTop;
				bvhNodeStack[stackTop].SetIdx(idx2);
				bvhNodeStack[stackTop].SetIsLeaf(isLeaf2);

				// if (stackTop < stackSize)
				// {
				// 	bvhNodeStack[stackTop] = idx2;
				// 	isLeaf[stackTop]       = isLeaf2;
				// }
				// else
				// {
				// 	Print("idx_error_stack3", Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y));
				// 	break;
				// }

				currIdx                = idx1;
				isCurrLeaf             = isLeaf1;

				//DEBUG_PRINT(stackTop);
				//DEBUG_PRINT(idx2);
				//DEBUG_PRINT(idx1);
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

