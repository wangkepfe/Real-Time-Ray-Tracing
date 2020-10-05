#pragma once

#include "kernel.cuh"
#include "bsdf.cuh"

// generate ray
__device__ void GenerateRay(
	Float3& orig,
	Float3& dir,
	Camera camera,
	Int2   idx,
	Float2 randPixelOffset,
	Float2 randAperture)
{
	// [0, 1] coordinates
	Float2 uv = (Float2(idx.x, idx.y) + randPixelOffset) * camera.inversedResolution;

	// [0, 1] -> [1, -1], since left/up vector should be 1 when uv is 0
	uv = uv * -2.0 + 1.0;

	// Point on the image plane
	Float3 pointOnImagePlane = camera.adjustedFront + camera.adjustedLeft * uv.x + camera.adjustedUp * uv.y;

	// Point on the aperture
	Float2 diskSample = ConcentricSampleDisk(randAperture);
	Float3 pointOnAperture = diskSample.x * camera.apertureLeft + diskSample.y * camera.apertureUp;

	// ray
	orig = camera.pos + pointOnAperture;
	dir = normalize(pointOnImagePlane - pointOnAperture);
}