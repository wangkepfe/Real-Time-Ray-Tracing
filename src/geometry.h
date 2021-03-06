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

	__host__ __device__ AABB() : max(), min() {}
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

struct __align__(16) Triangle
{
	__host__ __device__ Triangle() {}
	__host__ __device__ Triangle(const Float3& v1, const Float3& v2, const Float3& v3) : v1(v1), v2(v2), v3(v3) {}

	Float3 v1; float w1;
	Float3 v2; float w2;
	Float3 v3; float w3;
	Float3 v4; float w4;

	Float3 n1; float u1;
	Float3 n2; float u2;
	Float3 n3; float u3;
};

struct __align__(16) AABBCompact
{
	__host__ __device__ AABBCompact() {}
	__host__ __device__ AABBCompact(const AABB& aabb1, const AABB& aabb2)
	: box1maxX (aabb1.max.x), box1maxY (aabb1.max.y), box1minX (aabb1.min.x), box1minY (aabb1.min.y),
	  box2maxX (aabb2.max.x), box2maxY (aabb2.max.y), box2minX (aabb2.min.x), box2minY (aabb2.min.y),
	  box1maxZ (aabb1.max.z), box1minZ (aabb1.min.z), box2maxZ (aabb2.max.z), box2minZ (aabb2.min.z) {}

	__host__ __device__ AABB GetMerged() const {
		return AABB(Float3(max(box1maxX, box2maxX), max(box1maxY, box2maxY), max(box1maxZ, box2maxZ)),
		            Float3(min(box1minX, box2minX), min(box1minY, box2minY), min(box1minZ, box2minZ)));
	}

	__host__ __device__ AABB GetLeftAABB() const {
		return AABB(Float3(box1maxX, box1maxY, box1maxZ), Float3(box1minX, box1minY, box1minZ));
	}

	__host__ __device__ AABB GetRightAABB() const  {
		return AABB(Float3(box2maxX, box2maxY, box2maxZ), Float3(box2minX, box2minY, box2minZ));
	}

	float box1maxX;
	float box1maxY;
	float box1maxZ;
	float box2minX;

	float box1minX;
	float box1minY;
	float box1minZ;
	float box2minY;

	float box2maxX;
	float box2maxY;
	float box2maxZ;
	float box2minZ;
};