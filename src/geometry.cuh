#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"
#include "geometry.h"
#include "debug_util.cuh"

#define RayMax 10e10f

#define RAY_TRIANGLE_MOLLER_TRUMBORE 0
#define RAY_TRIANGLE_COORDINATE_TRANSFORM 1

#define RAY_TRIANGLE_CULLING 0
#define PRE_CALU_TRIANGLE_COORD_TRANS_OPT 0

// ------------------------------ Machine Epsilon -----------------------------------------------
// The smallest number that is larger than one minus one. ULP (unit in the last place) of 1
// ----------------------------------------------------------------------------------------------
__device__ __inline__ constexpr float MachineEpsilon()
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
__device__ __inline__ constexpr float ErrGamma(int n)
{
	return (n * MachineEpsilon()) / (1.0f - n * MachineEpsilon());
}

// ------------------------------ Sphere Ray Intersect ------------------------------------
// ray/sphere intersection. returns distance, RayMax if no intersection.
// ----------------------------------------------------------------------------------------
__device__ __inline__ float SphereRayIntersect(
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
__device__ __inline__ Float3 GetSphereRayIntersectPoint(
	const Sphere& sphere,
	const Ray&    ray,
	float         t,
	float&        pError)
{
	Float3 p                    = ray.orig + t * ray.dir;
	Float3 pRelative            = p - sphere.center;
	Float3 pRelativeReprojected = pRelative * sphere.radius / pRelative.length();
	Float3 pReprojected         = pRelativeReprojected + sphere.center;
	pError                      = abs(pReprojected).getmax() * ErrGamma(5);
	return pReprojected;
}

// ------------------------------ Aabb Ray Intersect ------------------------------
//	ray AABB intersect, return distance, RayMax if no hit
// --------------------------------------------------------------------------------
__device__ __inline__ float AabbRayIntersect(
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
__device__ __inline__ void GetAabbRayIntersectPointNormal(
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

// --------------------------- Get Aabb Ray Intersect Point Normal UV -------------------------
//	Get Aabb Ray Intersect reprojected Point and Normal and uv
// -----------------------------------------------------------------------------------------
__device__ __inline__ void GetAabbRayIntersectPointNormalUv(
	const AABB& aabb,
	const Ray&  ray,
	float       t,
	Float3&     point,
	Float3&     normal,
	Float2&     uv,
	float&      pError)
{
	Float3 p      = ray.orig + ray.dir * t;

	Float3 center = (aabb.max + aabb.min) / 2.0;
	Float3 v2     = aabb.max - aabb.min;
	Float3 v      = v2 / 2.0;

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
		//uv     = Float2((point.y - aabb.min.y) / v2.y, (point.z - aabb.min.z) / v2.z);
		uv     = Float2(point.y, point.z);
		pError = abs(point.x) * ErrGamma(5);
	}
	else if (maxDidx == 1)
	{
		normal = Float3(0, signD.y, 0);
		point  = Float3(p.x, center.y + signD.y * v.y, p.z);
		//uv     = Float2((point.x - aabb.min.x) / v2.x, (point.z - aabb.min.z) / v2.z);
		uv     = Float2(point.x, point.z);
		pError = abs(point.y) * ErrGamma(5);
	}
	else
	{
		normal = Float3(0, 0, signD.z);
		point  = Float3(p.x, p.y, center.z + signD.z * v.z);
		//uv     = Float2((point.x - aabb.min.x) / v2.x, (point.y - aabb.min.y) / v2.y);
		uv     = Float2(point.x, point.y);
		pError = abs(point.z) * ErrGamma(5);
	}
}

// --------------------------- Ray Plane Intersect -------------------------
// ray plane intersect, return distance, RayMax if no hit
// The plane equation is n dot p + w = 0
// -------------------------------------------------------------------------
__device__ __inline__ float RayPlaneIntersect( // N dot P = w
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
__device__ __inline__ Float3 GetRayPlaneIntersectPoint(
	const Float4& plane,
	const Ray&    ray,
	float         t,
	float&        pError)
{
	Float3 p  = ray.orig + ray.dir * t;
	Float3 pc = p - (dot(plane.xyz, p) + plane.w) * plane.xyz;
	pError    = abs(pc).getmax() * ErrGamma(6);
	return pc;
}

// ------------------------------ Moller Trumbore ----------------------------------------------------
// Ray Triangle intersection without pre-transformation
// --------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool MollerTrumbore(const Float3 &orig, const Float3 &dir, const Float3 &v0, const Float3 &v1, const Float3 &v2, float &t, float &u, float &v)
{
	Float3 v0v1 = v1 - v0;
	Float3 v0v2 = v2 - v0;
	Float3 pvec = cross(dir, v0v2);
	float det = dot(v0v1, pvec);

#if RAY_TRIANGLE_CULLING
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < ErrGamma(8)) return false;
#else
	// ray and triangle are parallel if det is close to 0
	if (abs(det) < ErrGamma(8)) return false;
#endif

	float invDet = 1 / det;

	Float3 tvec = orig - v0;
	u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	Float3 qvec = cross(tvec, v0v1);
	v = dot(dir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t = dot(v0v2, qvec) * invDet;
	if (t < ErrGamma(15)) return false;

	return true;
}

// ------------------------------ Ray Triangle Coordinate Transform ----------------------------------------------------
// Ray Triangle intersection with pre-transformation
// --------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool RayTriangleCoordinateTransform(Ray ray, Triangle triangle, float tCurrentHit, float& t, float& u, float& v, float& e)
{
	float origZ = triangle.w1 - dot(ray.orig, triangle.v1);
	float inverseDirZ = 1.0f / dot(ray.dir, triangle.v1);
	t = origZ * inverseDirZ;
	e = abs(t) * ErrGamma(7);

	if (t > ErrGamma(7) && t < tCurrentHit)
	{
		float origX = triangle.w2 + dot(ray.orig, triangle.v2);
		float dirX = dot(ray.dir, triangle.v2);
		u = origX + t * dirX;

		if (u >= 0.0f && u <= 1.0f)
		{
			float origY = triangle.w3 + dot(ray.orig, triangle.v3);
			float dirY = dot(ray.dir, triangle.v3);
			v = origY + t * dirY;

			if (v >= 0.0f && u + v <= 1.0f)
			{
				return true;
			}
		}
	}
	return false;
}

//-------------------------- Pre Calc Triangle Coord Trans -----------------------------
// Inverse the transform matrix
// [ E1x E2x a v1x ]
// [ E1y E2y b v1y ]
// [ E1z E2z c v1z ]
// [ 0   0   0  1  ]
//   Select f = (a,b,c) = (1,0,0) or (0,1,0) or (0,0,1)
// based on largest of (E1 x E2).xyz
// so that the matrix has a stable inverse
//-------------------------------------------------------
__device__ __forceinline__ void PreCalcTriangleCoordTrans(Triangle& triangle)
{
	Float3 v1 = triangle.v1;
	Float3 v2 = triangle.v2;
	Float3 v3 = triangle.v3;

	Float3 e1 = v2 - v1;
	Float3 e2 = v3 - v1;

	Float3 n = cross(e1, e2);

#if PRE_CALU_TRIANGLE_COORD_TRANS_OPT
	Float3 n_abs = abs(n);
	if (n_abs.x > n_abs.y && n_abs.x > n_abs.z)
	{
		// free vector (1, 0, 0)
		triangle.v1 = { 1           , n.y / n.x   , n.z / n.x };     triangle.w1 = dot(n, v1) / n.x;     // row3
		triangle.v2 = { 0           , e2.z / n.x  , -e2.y / n.x };  triangle.w2 = cross(v3, v1).x / n.x; // row1
		triangle.v3 = { 0           , -e1.z / n.x, e1.y / n.x };    triangle.w3 = -cross(v2, v1).x / n.x; // row2
	}
	else if (n_abs.y > n_abs.x && n_abs.y > n_abs.z)
	{
		// free vector (0, 1, 0)
		triangle.v1 = { n.x / n.y   , 1           , n.z / n.y };     triangle.w1 = dot(n, v1) / n.y;     // row3
		triangle.v2 = { -e2.z / n.y, 0           , e2.x / n.y };    triangle.w2 = cross(v3, v1).y / n.y; // row1
		triangle.v3 = { e1.z / n.y  , 0           , -e1.x / n.y };  triangle.w3 = -cross(v2, v1).y / n.y; // row2
	}
	else
	{
		// free vector (0, 0, 1)
		triangle.v1 = { n.x / n.z   , n.y / n.z   , 1 };             triangle.w1 = dot(n, v1) / n.z;     // row3
		triangle.v2 = { e2.y / n.z  , -e2.x / n.z, 0 };             triangle.w2 = cross(v3, v1).z / n.z; // row1
		triangle.v3 = { -e1.y / n.z, e1.x / n.z  , 0 };             triangle.w3 = -cross(v2, v1).z / n.z; // row2
	}
#else
	Mat4 mtx;

	mtx.setCol(0, { e1, 0 });
	mtx.setCol(1, { e2, 0 });
	mtx.setCol(2, { n, 0 });
	mtx.setCol(3, { v1, 1 });

	mtx = invert(mtx);

	triangle.v1 = mtx.getRow(2).xyz; triangle.w1 = - mtx.getRow(2).w;
	triangle.v2 = mtx.getRow(0).xyz; triangle.w2 = mtx.getRow(0).w;
	triangle.v3 = mtx.getRow(1).xyz; triangle.w3 = mtx.getRow(1).w;
	triangle.v4 = v1;
#endif
}

__device__ __forceinline__ float RayTriangleIntersect(Ray ray, Triangle triangle, float tCurr, float& u, float& v, float& e)
{
	float t;
#if RAY_TRIANGLE_MOLLER_TRUMBORE
	bool hit = MollerTrumbore(ray.orig, ray.dir, triangle.v1, triangle.v2, triangle.v3, t, u, v);
	if (hit == false) { return RayMax; }
	e = ErrGamma(15) * abs(t);
	return t;
#endif

#if RAY_TRIANGLE_COORDINATE_TRANSFORM
	bool hit = RayTriangleCoordinateTransform(ray, triangle, tCurr, t, u, v, e);
	if (hit == false) { return RayMax; }
	return t;
#endif
}

__device__ __forceinline__ bool RayAABBIntersect(const Float3& invRayDir, const Float3& rayOrig, const AABB& aabb, float& tmin, float& tmax) {

	Float3 t0s = (aabb.min - rayOrig) * invRayDir;
  	Float3 t1s = (aabb.max - rayOrig) * invRayDir;

  	Float3 tsmaller = min3f(t0s, t1s);
    Float3 tbigger  = max3f(t0s, t1s);

    tmin = tsmaller.getmax();
	tmax = tbigger.getmin();

	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	//if (IS_DEBUG_PIXEL())
	//{
		//printf("RayAABBIntersect: t0s=(%f, %f, %f), t1s=(%f, %f, %f)\n", t0s.x, t0s.y, t0s.z, t1s.x, t1s.y, t1s.z);
		//printf("RayAABBIntersect: tsmaller=(%f, %f, %f), tbigger=(%f, %f, %f)\n", tsmaller.x, tsmaller.y, tsmaller.z, tbigger.x, tbigger.y, tbigger.z);
		//printf("RayAABBIntersect: tmin=%f, tmax=%f\n", tmin, tmax);
	//}

	return (tmin < tmax) && (tmin > 0);
}

__device__ __forceinline__ void RayAabbPairIntersect(const Float3& invRayDir, const Float3& rayOrig, const AABBCompact& aabbpair, bool& intersect1, bool& intersect2, bool& isClosestIntersect1)
{
	AABB aabbLeft = aabbpair.GetLeftAABB();
	AABB aabbRight = aabbpair.GetRightAABB();

	float tmin1, tmin2;
	float tmax1, tmax2;

	intersect1 = RayAABBIntersect(invRayDir, rayOrig, aabbLeft, tmin1, tmax1);
	intersect2 = RayAABBIntersect(invRayDir, rayOrig, aabbRight, tmin2, tmax2);

	isClosestIntersect1 = (tmin1 <= tmin2);

	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	//if (IS_DEBUG_PIXEL())
	//{
	//	printf("RayAabbPairIntersect: aabbLeft.min=(%f, %f, %f), aabbLeft.max=(%f, %f, %f)\n", aabbLeft.min.x, aabbLeft.min.y, aabbLeft.min.z, aabbLeft.max.x, aabbLeft.max.y, aabbLeft.max.z);
	//	printf("RayAabbPairIntersect: aabbRight.min=(%f, %f, %f), aabbRight.max=(%f, %f, %f)\n", aabbRight.min.x, aabbRight.min.y, aabbRight.min.z, aabbRight.max.x, aabbRight.max.y, aabbRight.max.z);
	//	printf("RayAabbPairIntersect: intersect1=%d, intersect2=%d, isClosestIntersect1=%d\n", intersect1, intersect2, isClosestIntersect1);
	//}
}