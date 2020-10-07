#pragma once

#include "kernel.cuh"
#include "geometry.cuh"

// update material info after intersect
__device__ inline void UpdateMaterial(
	const ConstBuffer&   cbo,
	RayState&            rayState,
	const SceneMaterial& sceneMaterial)
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
			rayState.matType = PERFECT_REFLECTION;
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
#define RAY_TRAVERSE_AABBS 1
#define RAY_TRAVERSE_PLANE 1

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
	int&    objectIdx        = rayState.objectIdx;
	float&  rayOffset        = rayState.offset;
	bool&   isRayIntoSurface = rayState.isRayIntoSurface;
	float&  normalDotRayDir  = rayState.normalDotRayDir;
	Float2& uv               = rayState.uv;

	// init t with max distance
	float t         = RayMax;

	// init ray state
	objectIdx       = 0;
	intersectNormal = Float3(0);
	intersectPoint  = Float3(RayMax);
	uv              = Float2(0);

#if RAY_TRAVERSE_SPHERES
	// ----------------------- spheres ---------------------------
	// get spheres
	int numSpheres  = sceneGeometry.numSpheres;
	Sphere* spheres = sceneGeometry.spheres;

	// init error
	rayOffset    = 1e-7f;
	float errorP = 1e-7f;
	float errorT = 1e-7f;

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
			objectIdx       = i;
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
	Float3 oneOverRayDir = SafeDivede3f(Float3(1.0f), ray.dir);
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
	if (t_temp < t)
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

	UpdateMaterial(cbo, rayState, sceneMaterial);
}
