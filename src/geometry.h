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