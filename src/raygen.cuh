#pragma once

#include "kernel.cuh"
#include "bsdf.cuh"

// generate ray
__device__ void GenerateRay(
	Float3& orig,
	Float3& dir,
	Camera camera,
	Int2   idx,
	Float2 randNum1,
	Float2 randNum2)
{
	// [0, 1] coordinates
	Float2 xy = Float2(idx.x, idx.y) + randNum1;
	Float2 uv = xy * Float2(camera.resolution.z, camera.resolution.w);

	// [0, 1] -> [1, -1]
	uv = uv * -2.0 + 1.0;

	// left vector and up vector
	Float3 left = camera.leftAperture.xyz * uv.x * camera.fov.z;
	Float3 up   = camera.up.xyz * uv.y * camera.fov.w;

	// Point on the image plane
	Float3 pointOnImagePlane = (camera.dirFocal.xyz + left + up) * camera.dirFocal.w;

	// Point on the aperture
	Float2 diskSample = ConcentricSampleDisk(randNum2);
	Float3 pointOnAperture = (diskSample.x * camera.leftAperture.xyz + diskSample.y * camera.up.xyz) * camera.leftAperture.w;

	// ray origin
	orig = camera.pos.xyz + pointOnAperture;

	// ray dir
	dir = normalize(pointOnImagePlane - pointOnAperture);
}