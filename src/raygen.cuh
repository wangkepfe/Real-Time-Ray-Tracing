#pragma once

#include "kernel.cuh"
#include "bsdf.cuh"

// generate ray
__device__ void inline GenerateRay(
	Float3& orig,
	Float3& dir,
	Float3& centerDir,
	Float2& sampleUv,
	Camera camera,
	Int2   idx,
	Float2 randPixelOffset,
	Float2 randAperture)
{
	// [0, 1] coordinates
	Float2 uv = (Float2(idx.x, idx.y) + randPixelOffset) * camera.inversedResolution;
	Float2 uvCenter = (Float2(idx.x, idx.y) + 0.5f) * camera.inversedResolution;
	sampleUv = uv; // Float2(idx.x, idx.y) * camera.inversedResolution;

	// [0, 1] -> [1, -1], since left/up vector should be 1 when uv is 0
	uv = uv * -2.0f + 1.0f;
	uvCenter = uvCenter * -2.0f + 1.0f;

	// Point on the image plane
	Float3 pointOnImagePlane = camera.adjustedFront + camera.adjustedLeft * uv.x + camera.adjustedUp * uv.y;
	Float3 pointOnImagePlaneCenter = camera.adjustedFront + camera.adjustedLeft * uvCenter.x + camera.adjustedUp * uvCenter.y;

	// Point on the aperture
	Float2 diskSample = ConcentricSampleDisk(randAperture);
	Float3 pointOnAperture = diskSample.x * camera.apertureLeft + diskSample.y * camera.apertureUp;

	// ray
	orig = camera.pos + pointOnAperture;
	dir = normalize(pointOnImagePlane - pointOnAperture);
	centerDir = normalize(pointOnImagePlaneCenter);
}

__device__ Float2 inline copysignf2(Float2 a, Float2 b)
{
	return { copysignf(a.x, b.x), copysignf(a.y, b.y) };
}

__device__ float inline GetRayConeWidth(Camera camera, Int2 idx)
{
	Float2 pixelCenter = (Float2(idx.x, idx.y) + 0.5f) - Float2(camera.resolution.x, camera.resolution.y) / 2;
	Float2 pixelOffset = copysignf2(Float2(0.5f), pixelCenter);

	Float2 uvNear = (pixelCenter - pixelOffset) * camera.inversedResolution * 2; // [-1, 1]
	Float2 uvFar = (pixelCenter + pixelOffset) * camera.inversedResolution * 2;

	Float2 halfFovLength = Float2(tanf(camera.fov.x / 2), tanf(camera.fov.y / 2));

	Float2 pointOnPlaneNear = uvNear * halfFovLength;
	Float2 pointOnPlaneFar = uvFar * halfFovLength;

	float angleNear = atanf(pointOnPlaneNear.length());
	float angleFar = atanf(pointOnPlaneFar.length());

	float pixelAngleWidth = angleFar - angleNear;

	return pixelAngleWidth;
}