#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"
#include "geometry.h"

#define RayMax 10e10f

// ------------------------------ Machine Epsilon -----------------------------------------------
// The smallest number that is larger than one minus one. ULP (unit in the last place) of 1
// ----------------------------------------------------------------------------------------------
__device__ constexpr float MachineEpsilon()
{
	typedef union {
		float f32;
		int i32;
	} flt_32;

	flt_32 s{ 1.0f };

	s.i32++;
	return (s.f32 - 1.0f);
}

// ------------------------------ Error Gamma -------------------------------------------------------
// return 32bit floating point arithmatic calculation error upper bound, n is number of calculation
// --------------------------------------------------------------------------------------------------
__device__ constexpr float ErrGamma(int n)
{
	return (n * MachineEpsilon()) / (1.0f - n * MachineEpsilon());
}

// ------------------------------ Sphere Ray Intersect ------------------------------------
// ray/sphere intersection. returns distance, RayMax if no intersection.
// ----------------------------------------------------------------------------------------
__device__ float SphereRayIntersect(
	const Sphere& sphere,
	const Ray&    ray,
	float&        tError)
{
	Float3 L   = sphere.center - ray.orig;
	float b    = dot(L, ray.dir);
	float b2   = b * b;
	float L2   = dot(L, L);
	float r2   = sphere.radius * sphere.radius;
	float c    = L2 - r2;
	float disc = b2 - c;

	if (disc < 0)
		return RayMax;

	disc = sqrtf(disc);

	float t1 = (b > 0) ? (b + disc) : (b - disc);
	float t2 = c / t1;

	if (t2 < t1) swap(t1, t2);

	tError = max1f(abs(t1) * ErrGamma(7), abs(t2) * ErrGamma(8));

	if (t1 > tError)
		return t1;
	else if (t2 > tError)
		return t2;
	else
		return RayMax;
}

// ------------------------------ Get Sphere Ray Intersect Point --------------------------
// reproject intersection point to increase accuracy
// ----------------------------------------------------------------------------------------
__device__ Float3 GetSphereRayIntersectPoint(
	const Sphere& sphere,
	const Ray&    ray,
	float         t,
	float&        pError)
{
	Float3 p                    = ray.orig + t * ray.dir;
	Float3 pRelative            = p - sphere.center;
	Float3 pRelativeReprojected = pRelative * sphere.radius / pRelative.length();
	Float3 pReprojected         = pRelativeReprojected + sphere.center;
	pError                      = abs(pReprojected).max() * ErrGamma(5);
	return pReprojected;
}

// ------------------------------ Aabb Ray Intersect ------------------------------
//	ray AABB intersect, return distance, RayMax if no hit
// --------------------------------------------------------------------------------
__device__ float AabbRayIntersect(
	const Float3& aabbMin,
	const Float3& aabbMax,
	const Float3& rayOrig,
	const Float3& oneOverRayDir,
	const Float3& rayOrigOverRayDir,
	float&        tError)
{
	float t1 = aabbMin.x * oneOverRayDir.x - rayOrigOverRayDir.x;
	float t2 = aabbMax.x * oneOverRayDir.x - rayOrigOverRayDir.x;
	float t3 = aabbMin.y * oneOverRayDir.y - rayOrigOverRayDir.y;
	float t4 = aabbMax.y * oneOverRayDir.y - rayOrigOverRayDir.y;
	float t5 = aabbMin.z * oneOverRayDir.z - rayOrigOverRayDir.z;
	float t6 = aabbMax.z * oneOverRayDir.z - rayOrigOverRayDir.z;

	float tmin = max1f(max1f(min1f(t1, t2), min1f(t3, t4)), min1f(t5, t6));
	float tmax = min1f(min1f(max1f(t1, t2), max1f(t3, t4)), max1f(t5, t6));

	float error_tmax = abs(tmax) * ErrGamma(3);
	float error_tmin = abs(tmin) * ErrGamma(3);

	// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
	// if tmin > tmax, ray doesn't intersect AABB
	if (tmax < error_tmax || tmin > tmax)
	{
		tError = error_tmax;
		return RayMax;
	}

	// if tmin < 0 then the ray origin is inside of the AABB and tmin is behind the start of the ray so tmax is the first intersection
	if (tmin < error_tmin)
	{
		tError = error_tmax;
		return tmax;
	}
	else
	{
		tError = error_tmin;
		return tmin;
	}
}

// --------------------------- Get Aabb Ray Intersect Point Normal -------------------------
//	Get Aabb Ray Intersect reprojected Point and Normal
// -----------------------------------------------------------------------------------------
__device__ void GetAabbRayIntersectPointNormal(
	const AABB& aabb,
	const Ray&  ray,
	float       t,
	Float3&     point,
	Float3&     normal,
	float&      pError)
{
	Float3 p      = ray.orig + ray.dir * t;

	Float3 center = (aabb.max + aabb.min) / 2.0;
	Float3 v      = (aabb.max - aabb.min) / 2.0;

	Float3 D      = p - center;
	Float3 absD   = abs(D);
	Float3 signD  = D / absD;

	int maxDidx = 0;
	if (absD.x >= absD.y) // x y
	{
		maxDidx = (absD.x >= absD.z) ? 0 : 2; // x yz, z x y
	}
	else // y x
	{
		maxDidx = (absD.y >= absD.z) ? 1 : 2; // y xz, z y x
	}

	if (maxDidx == 0)
	{
		normal = Float3(signD.x, 0, 0);
		point  = Float3(center.x + signD.x * v.x, p.y, p.z);
		pError = abs(point.x) * ErrGamma(5);
	}
	else if (maxDidx == 1)
	{
		normal = Float3(0, signD.y, 0);
		point  = Float3(p.x, center.y + signD.y * v.y, p.z);
		pError = abs(point.y) * ErrGamma(5);
	}
	else
	{
		normal = Float3(0, 0, signD.z);
		point  = Float3(p.x, p.y, center.z + signD.z * v.z);
		pError = abs(point.z) * ErrGamma(5);
	}
}

// --------------------------- Ray Plane Intersect -------------------------
// ray plane intersect, return distance, RayMax if no hit
// -------------------------------------------------------------------------
__device__ float RayPlaneIntersect(
	const Float4& plane,
	const Ray&    ray,
	float&        tError)
{
	Float3 planeNormal = -plane.xyz;
    float denom        = dot(planeNormal, ray.dir);

    if (denom > 1e-7f) // ray plane parallel
	{
        Float3 rayOrigToPlanePoint = planeNormal * plane.w - ray.orig;
        float t                    = dot(rayOrigToPlanePoint, planeNormal) / denom;

		tError                     = abs(t) * ErrGamma(6);

		if (t >= tError)
		{
			return t;
		}
    }

	return RayMax;
}

// --------------------------- Get Ray Plane Intersect Point -------------------------
//	Get plane, Ray Intersect reprojected Point
// -----------------------------------------------------------------------------------
__device__ Float3 GetRayPlaneIntersectPoint(
	const Float4& plane,
	const Ray&    ray,
	float         t,
	float&        pError)
{
	Float3 p  = ray.orig + ray.dir * t;
	Float3 pc = p - (dot(plane.xyz, p) + plane.w) * plane.xyz;
	pError    = abs(pc).max() * ErrGamma(6);
	return pc;
}