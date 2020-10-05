#pragma once

#include "linear_math.h"

struct  __align__(16) Sphere
{
	Float3 center;
	float  radius;

	__host__ __device__ Sphere() : center(0), radius(0) {}
	__host__ __device__ Sphere(Float3 center, float radius) : center{ center }, radius{ radius }{}
};

struct  __align__(16) AABB
{
	Float3 max;
	float pad16;

	Float3 min;
	float pad32;

	__host__ __device__ AABB() {}
	__host__ __device__ AABB(const Float3& max, const Float3& min) : max{ max }, min{ min } {}
	__host__ __device__ AABB(const Float3& center, float edgeLength) : max{ center + Float3(edgeLength / 2.0f) }, min{ center - Float3(edgeLength / 2.0f) } {}
};

struct  __align__(16) Ray
{
	Float3 orig;
	float pad16;

	Float3 dir;
	float pad32;

	__host__ __device__ Ray() : orig(0), dir(0) {}
	__host__ __device__ Ray(Float3 orig, Float3 dir) : orig{ orig }, dir{ dir }{}
};

struct __align__(16) AabbPair
{
	union {
		struct {
			Float4 n0xy; // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			Float4 n1xy; // childnode 1, xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			Float4 nz;   // childnode 0 and 1, z-bounds (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
		};
		Float4 v[3];
	};

	__host__ __device__ AabbPair() : n0xy(), n1xy(), nz() {}
};

struct __align__(16) Triangle
{
	union
	{
		struct { Float4 p0, p1, p2; };
		Float4 v[3];
	};

	__host__ __device__ Triangle() : v{} {}
	__host__ __device__ Triangle(const Float3& p0, const Float3& p1, const Float3& p2) : p0{p0}, p1{p1}, p2{p2} {}


	// __host__ __device__ void PreTransform()
	// {
	// 	const Float3& v0 = v[0].xyz;
	// 	const Float3& v1 = v[1].xyz;
	// 	const Float3& v2 = v[2].xyz;

	// 	Mat4 mtx;
	// 	mtx.setCol(0, Float4(v0 - v2, 0.0f));
	// 	mtx.setCol(1, Float4(v1 - v2, 0.0f));
	// 	mtx.setCol(2, Float4(cross(v0 - v2, v1 - v2), 0.0f));
	// 	mtx.setCol(3, Float4(v2, 1.0f));
	// 	mtx = mtx.invert();

	// 	v[0] = Float4(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3));
	// 	v[1] = mtx.getRow(0);
	// 	v[2] = mtx.getRow(1);
	// }
};