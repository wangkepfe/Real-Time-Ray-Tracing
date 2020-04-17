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
	float&               rayOffset)          // [out]
{
	// init t with max distance
	float t = RayMax;
	rayOffset = 1e-7f;
	objectIdx = 0;
	intersectNormal = Float3(0, 1, 0);
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

    // plane
    Float4 planes[1];
    planes[0] = Float4(0, 1, 0, 0);

    for (int i = 0; i < 1; ++i)
    {
        Float4& plane = planes[i];

        if (dot(plane.xyz, ray.dir) < 0)
        {
            float t_temp = RayPlaneIntersect(plane, ray, errorT);

            if (t_temp < t)
            {
                t               = t_temp;
                objectIdx       = 998;
                intersectPoint  = GetRayPlaneIntersectPoint(plane, ray, t, errorP);
                intersectNormal = plane.xyz;
                rayOffset       = errorT + errorP;
            }
        }
    }

	// return true if hit
	return (t < RayMax);
}
