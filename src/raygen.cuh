#pragma once

#include "kernel.cuh"

// generate ray
__device__ Ray GenerateRay(
	Camera camera,
	Int2   idx,
	Float2 randNum)
{
	Ray ray;

	// ray origin
	ray.orig = camera.pos.xyz;

	// [0, 1] coordinates
	Float2 xy = Float2(idx.x, idx.y);
	xy = xy + randNum;
	Float2 uv = xy * Float2(camera.resolution.z, camera.resolution.w);

	// [0, 1] -> [1, -1]
	uv = uv * -2.0 + 1.0;

	// left vector and up vector
	Float3 left = camera.left.xyz * uv.x * camera.fov.z;
	Float3 up = Float3(0, 1, 0) * uv.y * camera.fov.w;

	// ray direction
	ray.dir = camera.dir.xyz + left + up;

	// normalize ray direction
	ray.dir.normalize();

	return ray;
}