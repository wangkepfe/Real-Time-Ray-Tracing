#pragma once

#include <cuda_runtime.h>
#include "linearMath.h"
#include "geometry.h"
#include "debugUtil.h"
#include "precision.cuh"

#define RAY_TRIANGLE_MOLLER_TRUMBORE 0
#define RAY_TRIANGLE_COORDINATE_TRANSFORM 0
#define RAY_TRIANGLE_WATERTIGHT 1

#define RAY_TRIANGLE_CULLING 0

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
__host__ __device__ inline Float3 GetRayPlaneIntersectPoint(
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
__host__ __device__ inline bool MollerTrumbore(const Float3 &orig, const Float3 &dir, const Float3 &v0, const Float3 &v1, const Float3 &v2, float &t, float &u, float &v)
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
__host__ __device__ __forceinline__ bool RayTriangleCoordinateTransform(Ray ray, Triangle triangle, float tCurrentHit, float& t, float& u, float& v, float& e)
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
__device__ __forceinline__ Triangle PreCalcTriangleCoordTrans(const Triangle& triangle)
{
	Float3 v1 = triangle.v1;
	Float3 v2 = triangle.v2;
	Float3 v3 = triangle.v3;

	Float3 e1 = v2 - v1;
	Float3 e2 = v3 - v1;

	Float3 n = cross(e1, e2);

	Mat4 mtx;

	mtx.setCol(0, { e1, 0 });
	mtx.setCol(1, { e2, 0 });
	mtx.setCol(2, { n, 0 });
	mtx.setCol(3, { v1, 1 });

	mtx = invert(mtx);

	Triangle transformedTriangle;

	transformedTriangle.v1 = mtx.getRow(2).xyz;
	transformedTriangle.v2 = mtx.getRow(0).xyz;
	transformedTriangle.v3 = mtx.getRow(1).xyz;

	transformedTriangle.w1 = - mtx.getRow(2).w;
	transformedTriangle.w2 = mtx.getRow(0).w;
	transformedTriangle.w3 = mtx.getRow(1).w;

	transformedTriangle.v4 = v1;

	return transformedTriangle;
}

__host__ __device__ __forceinline__ int max_dim(Float3 dir)
{
	if (dir.x > dir.y)
	{
		if (dir.z > dir.x) return 2;
		else return 0;
	}
	else
	{
		if (dir.z > dir.y) return 2;
		else return 1;
	}
}

__host__ __device__ __forceinline__ int sign_mask(float a)
{
	int b = *reinterpret_cast<int*>(&a);
	return b & 0x80000000;
}

__host__ __device__ __forceinline__ float xorf(float a, int b)
{
	int c = *reinterpret_cast<int*>(&a);
	c = c ^ b;
	float d = *reinterpret_cast<float*>(&c);
	return d;
}

// ------------------------------ Ray Triangle Watertight ----------------------------------------------------
// Ray Triangle intersection described in Watertight Ray/Triangle Intersection https://jcgt.org/published/0002/01/05/paper.pdf
// -----------------------------------------------------------------------------------------------------------
__host__ __device__ __forceinline__ bool RayTriangleWatertight(Ray ray, Triangle tri, float tCurrentHit, float& t, float& u, float& v, float& e)
{
	Float3 dir = ray.dir;
	Float3 org = ray.orig;

	int kz = max_dim(abs(dir));
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

	if (dir[kz] < 0.0f) swap(kx,ky);

	float Sx = dir[kx]/dir[kz];
	float Sy = dir[ky]/dir[kz];
	float Sz = 1.0f/dir[kz];

	const Float3 A = tri.v1-org;
	const Float3 B = tri.v2-org;
	const Float3 C = tri.v3-org;

	const float Ax = A[kx] - Sx*A[kz];
	const float Ay = A[ky] - Sy*A[kz];
	const float Bx = B[kx] - Sx*B[kz];
	const float By = B[ky] - Sy*B[kz];
	const float Cx = C[kx] - Sx*C[kz];
	const float Cy = C[ky] - Sy*C[kz];

	float U = Cx*By - Cy*Bx;
	float V = Ax*Cy - Ay*Cx;
	float W = Bx*Ay - By*Ax;

	if (U == 0.0f || V == 0.0f || W == 0.0f) {
		double CxBy = (double)Cx*(double)By;
		double CyBx = (double)Cy*(double)Bx;
		U = (float)(CxBy - CyBx);
		double AxCy = (double)Ax*(double)Cy;
		double AyCx = (double)Ay*(double)Cx;
		V = (float)(AxCy - AyCx);
		double BxAy = (double)Bx*(double)Ay;
		double ByAx = (double)By*(double)Ax;
		W = (float)(BxAy - ByAx);
	}

	if ((U<0.0f || V<0.0f || W<0.0f) && (U>0.0f || V>0.0f || W>0.0f))
		return false;

	float det = U+V+W;
	if (det == 0.0f)
		return false;

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if ((xorf(T,det_sign) < 0.0f) || (xorf(T,det_sign) > tCurrentHit * xorf(det, det_sign)))
		return false;

	const float rcpDet = 1.0f/det;
	u = U*rcpDet;
	v = V*rcpDet;
	t = T*rcpDet;

	e = ErrGamma(16) * abs(t);

	return true;
}

__host__ __device__ inline float RayTriangleIntersect(Ray ray, Triangle triangle, float tCurr, float& u, float& v, float& e)
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

#if RAY_TRIANGLE_WATERTIGHT
	bool hit = RayTriangleWatertight(ray, triangle, tCurr, t, u, v, e);
	if (hit == false) { return RayMax; }
	return t;
#endif
}

struct RayBoxIntersectionHelper
{
	int nearX;
	int nearY;
	int nearZ;
	int farX ;
	int farY ;
	int farZ ;
	float org_near_x;
	float org_near_y;
	float org_near_z;
	float org_far_x;
	float org_far_y;
	float org_far_z;
	float rdir_near_x;
	float rdir_near_y;
	float rdir_near_z;
	float rdir_far_x ;
	float rdir_far_y ;
	float rdir_far_z ;
};

__host__ __device__ inline  bool RayAABBIntersect(/*const Float3& invRayDir, const Float3& rayOrig,*/ const AABB& aabb, RayBoxIntersectionHelper helper, float& tNear, float& tFar) {

	float tNearX = (aabb[helper.nearX] - helper.org_near_x) * helper.rdir_near_x;
	float tNearY = (aabb[helper.nearY] - helper.org_near_y) * helper.rdir_near_y;
	float tNearZ = (aabb[helper.nearZ] - helper.org_near_z) * helper.rdir_near_z;
	float tFarX = (aabb[helper.farX] - helper.org_far_x ) * helper.rdir_far_x;
	float tFarY = (aabb[helper.farY] - helper.org_far_y ) * helper.rdir_far_y;
	float tFarZ = (aabb[helper.farZ] - helper.org_far_z ) * helper.rdir_far_z;

	tNear = max(tNearX,tNearY,tNearZ);
	tFar = min(tFarX ,tFarY ,tFarZ);

	bool hit = tNear <= tFar && (tFar > 0);
	tNear = max(tNear, 0.0f);

	return hit;

	// Float3 t0s = (aabb.min - rayOrig) * invRayDir;
  	// Float3 t1s = (aabb.max - rayOrig) * invRayDir;

  	// Float3 tsmaller = min3f(t0s, t1s);
    // Float3 tbigger  = max3f(t0s, t1s);

    // tmin = tsmaller.getmax();
	// tmax = tbigger.getmin();

	// #if DEBUG_RAY_AABB_INTERSECT_DETAIL
	// Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	// if (IS_DEBUG_PIXEL())
	// {
	// 	printf("RayAABBIntersect: t0s=(%f, %f, %f), t1s=(%f, %f, %f)\n", t0s.x, t0s.y, t0s.z, t1s.x, t1s.y, t1s.z);
	// 	printf("RayAABBIntersect: tsmaller=(%f, %f, %f), tbigger=(%f, %f, %f)\n", tsmaller.x, tsmaller.y, tsmaller.z, tbigger.x, tbigger.y, tbigger.z);
	// 	printf("RayAABBIntersect: tmin=%f, tmax=%f\n", tmin, tmax);
	// }
	// #endif

	// bool result = (tmin < tmax) && (tmax > 0);

	// tmin = max(tmin, 0.0f);

	// return result;
}

__host__ __device__ inline void RayAabbPairIntersect(RayBoxIntersectionHelper helper, const Float3& invRayDir, const Float3& rayOrig, const AABBCompact& aabbpair, bool& intersect1, bool& intersect2, float& t1, float& t2)
{
	AABB aabbLeft = aabbpair.GetLeftAABB();
	AABB aabbRight = aabbpair.GetRightAABB();

	float tmin1, tmin2;
	float tmax1, tmax2;

	intersect1 = RayAABBIntersect(/*invRayDir, rayOrig,*/  aabbLeft, helper,tmin1, tmax1);
	intersect2 = RayAABBIntersect(/*invRayDir, rayOrig,*/  aabbRight,helper, tmin2, tmax2);

	t1 = tmin1;
	t2 = tmin2;

	#if DEBUG_RAY_AABB_INTERSECT
	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (IS_DEBUG_PIXEL())
	{
		Float3 rayDir = SafeDivide3f(Float3(1.0f), invRayDir);
		DEBUG_PRINT(rayOrig)
		DEBUG_PRINT(rayDir)
		printf("RayAabbPairIntersect: aabbLeft.min=(%f, %f, %f), aabbLeft.max=(%f, %f, %f)\n", aabbLeft.min.x, aabbLeft.min.y, aabbLeft.min.z, aabbLeft.max.x, aabbLeft.max.y, aabbLeft.max.z);
		printf("RayAabbPairIntersect: aabbRight.min=(%f, %f, %f), aabbRight.max=(%f, %f, %f)\n", aabbRight.min.x, aabbRight.min.y, aabbRight.min.z, aabbRight.max.x, aabbRight.max.y, aabbRight.max.z);
		printf("RayAabbPairIntersect: intersect1=%d, intersect2=%d, isClosestIntersect1=%d\n", intersect1, intersect2, isClosestIntersect1);
	}
	#endif
}