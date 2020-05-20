#pragma once

#include "kernel.cuh"
#include "geometry.cuh"

// ray traverse, return true if hit
__device__ bool RaySceneIntersect(
	const Ray&           ray,                // [in]
	const SceneGeometry& sceneGeometry,      // [in]
	Float3&              intersectPoint,     // [out]
	Float3&              intersectNormal,    // [out]
	int&                 objectIdx,          // [out]
	float&               rayOffset,          // [out]
	float&               rayDist,
	bool&                isRayIntoSurface,
	float&               normalDotRayDir)
{
	// init t with max distance
	float t = RayMax;
	rayOffset = 1e-7f;
	objectIdx = 0;
	intersectNormal = Float3(0, 0, 0);
	intersectPoint = Float3(RayMax);

	// get spheres
	int numSpheres  = sceneGeometry.numSpheres;
	Sphere* spheres = sceneGeometry.spheres;
	float errorP = 10e-7f;
	float errorT = 10e-7f;

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

	// get aabb
	int numAabbs = sceneGeometry.numAabbs;
	AABB* aabbs = sceneGeometry.aabbs;
	Float3 oneOverRayDir = 1.0 / ray.dir;
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

			rayOffset       = errorT + errorP;
		}
	}

	rayDist += distance(intersectPoint, ray.orig);

	// Is ray into surface? If no, flip normal
	normalDotRayDir = dot(intersectNormal, ray.dir);    // ray dot geometry normal
	isRayIntoSurface = normalDotRayDir < 0;     // ray shoot into surface, if dot < 0
	if (isRayIntoSurface == false)
	{
		intersectNormal = -intersectNormal; // if ray not shoot into surface, convert the case to "ray shoot into surface" by flip the normal
		normalDotRayDir = -normalDotRayDir;
	}

	// return true if hit
	return (t < RayMax);
}
