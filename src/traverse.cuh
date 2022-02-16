#pragma once

#include "kernel.cuh"
#include "geometry.cuh"
#include "bvhNode.cuh"
#include "traverse.h"

// update material info after intersect
__device__ inline void UpdateMaterial(
	const ConstBuffer&   cbo,
	RayState&            rayState,
	const SceneMaterial& sceneMaterial,
	const SceneGeometry& sceneGeometry)
{
	// material type
	if (rayState.hit == false)
	{
		rayState.matType = MAT_SKY;
		rayState.matId = 99999;
	}
	else
	{
		rayState.matId = SAFE_LOAD(sceneMaterial.materialsIdx, rayState.objectIdx, sceneMaterial.numMaterialsIdx, 6);

		if (rayState.objectIdx == PLANE_OBJECT_IDX) // special case for floor, will be removed
		{
			rayState.matId = 6;
		}

		SurfaceMaterial mat = SAFE_LOAD(sceneMaterial.materials, rayState.matId, sceneMaterial.numMaterials, SurfaceMaterial{});
		rayState.matType = mat.type;
	}

	// shadow ray
	if (rayState.isShadowRay)
	{
		if ((rayState.matType == EMISSIVE && rayState.lightIdx == rayState.objectIdx) ||
		    (rayState.matType == MAT_SKY && rayState.lightIdx == ENV_LIGHT_ID))
		{
			rayState.hitLight = true;
		}
		else
		{
			rayState.isOccluded = true;
		}
	}
	else // non-shadow ray
	{
		rayState.hitLight = rayState.matType == EMISSIVE || rayState.matType == MAT_SKY;
	}

	// is diffuse
	rayState.isDiffuse = (rayState.matType == LAMBERTIAN_DIFFUSE) || (rayState.matType == MICROFACET_REFLECTION);
}

#define RAY_TRAVERSE_SPHERES 0
#define RAY_TRAVERSE_AABBS 0
#define RAY_TRAVERSE_PLANE 0
#define RAY_TRAVERSE_BVH 1

// ray traverse, return true if hit
__device__ inline void RaySceneIntersect(
	const ConstBuffer&   cbo,
	const SceneMaterial& sceneMaterial,
	const SceneGeometry& sceneGeometry,
	RayState&            rayState)
{
	if (rayState.hitLight == true || rayState.isHitProcessed == false || rayState.isOccluded == true) { return; }

	rayState.isHitProcessed = false;

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

	TraverseBvh(
		sceneGeometry.triangles,
		sceneGeometry.bvhNodes,
		sceneGeometry.tlasBvhNodes,
		cbo.bvhBatchSize,
		ray,
		invRayDir,
		errorP,
		errorT,
		t,
		uv,
		objectIdx,
		intersectNormal,
		fakeNormal,
		intersectPoint,
		rayOffset,
		cbo.bvhNodesSize,
		cbo.tlasBvhNodesSize,
		cbo.trianglesSize);

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
		#if USE_INTERPOLATED_FAKE_NORMAL
		fakeNormal = intersectNormal;
		#endif
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

	#if USE_INTERPOLATED_FAKE_NORMAL
	if (dot(fakeNormal, intersectNormal) < 0)
	{
		fakeNormal = -fakeNormal;
	}
	#endif

	rayState.hit = (t < RayMax);
	rayState.depth = t;

	if (rayState.hit == false)
	{
		rayState.normal = Float3(0, -1, 0);
		#if USE_INTERPOLATED_FAKE_NORMAL
		fakeNormal = Float3(0, -1, 0);
		#endif
	}

	UpdateMaterial(cbo, rayState, sceneMaterial, sceneGeometry);
}

