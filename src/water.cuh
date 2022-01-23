#pragma once

#include <cuda_runtime.h>
#include "geometry.cuh"
#include "debug_util.cuh"
#include "sampler.cuh"


inline __device__ float GetWaves(Float2 pos, int iterations, float clockTime)
{
	pos *= 0.15;

	float iter      = 0.0;
    float weight    = 1.0;
    float w         = 0.0;
    float ws        = 0.0;
	float speed     = 2.0;
	float frequency = 6.0;

	for(int i = 0; i < iterations; i++)
	{
        Float2 dir = Float2(sin(iter), cos(iter));
		float x    = dot(dir, pos) * frequency + clockTime * speed;
		float wave = expf(sinf(x) - 1.0);
		float dx   = -wave * cosf(x);
        pos       += normalize(dir) * dx * weight * 0.048;
        w         += wave * weight;
        ws        += weight;
		iter      += 12.0;
        weight    *= 0.8;
        frequency *= 1.18;
        speed     *= 1.07;
    }

    return w / ws;
}

#define WATER_GEOMETRY_QUALITY 8
#define WATER_GEOMETRY_TRAVERSE_STEP 3

inline __device__ void RayMarchWater(float& dist, Float3& hitpos, Float3 camera, Float3 start, Float3 end, float clockTime)
{
	int iter = WATER_GEOMETRY_QUALITY;

#if 1
	Float3 pos = start;
    Float3 vec = end - start;

	float t_high = 0;
	float t_low = 1;
	float t_mid;
	float h_high = start.y - (1.0 - GetWaves(start.xz(), iter, clockTime)) * vec.y;
	float h_low = end.y - (1.0 - GetWaves(end.xz(), iter, clockTime)) * vec.y;
	float h_mid = 0.0;

    for (int i = 0; i < WATER_GEOMETRY_TRAVERSE_STEP; i++)
	{
		t_mid = mix1f(t_high, t_low, h_high / (h_high - h_low));
		pos = start + vec * t_mid;
        h_mid = pos.y - (1.0 - GetWaves(pos.xz(), iter, clockTime)) * vec.y;

		if (h_mid < 0.0)
		{
        	t_low = t_mid;
            h_low = h_mid;
        }
		else
		{
            t_high = t_mid;
            h_high = h_mid;
        }
    }

	t_mid = mix1f(t_high, t_low, h_high / (h_high - h_low));
	hitpos = start + vec * t_mid;
	dist = distance(hitpos, camera);

#else
	Float3 pos = start;
	Float3 vec = end - start;
	float maxStep = 256;
	Float3 step = vec / maxStep;
	int i = 0;
	for (; i < maxStep; i++)
	{
		float waveHeight = GetWaves(pos.xz(), iter, clockTime);
		waveHeight = (1.0 - waveHeight) * vec.y;
		if (waveHeight > pos.y)
		{
			hitpos = pos;
			dist = distance(hitpos, camera);
			break;
		}
		pos += step;
	}
#endif
}

#define WATER_NORMAL_QUALITY 30

inline __device__ Float3 GetWaterNormal(Float2 pos, float depth, float clockTime, float distFallOff)
{
	const float eps = 1e-3;
	int iter = WATER_NORMAL_QUALITY;
	Float3 normal;
#if 0
	Float3 n;
    n.y = GetWaves(pos, iter, clockTime) * depth;
    n.x = GetWaves(pos + Float2(eps, 0), iter, clockTime) * depth - n.y;
    n.z = GetWaves(pos + Float2(0, eps), iter, clockTime) * depth - n.y;
    n.y = eps;
	normal = normalize(n);
#else
	Float2 p = pos;
	Float3 a = Float3(p.x, GetWaves(p, iter, clockTime) * depth, p.y);
	p = Float2(pos.x + eps, pos.y);
	Float3 b = Float3(p.x, GetWaves(p, iter, clockTime) * depth, p.y);
	p = Float2(pos.x, pos.y + eps);
	Float3 c = Float3(p.x, GetWaves(p, iter, clockTime) * depth, p.y);
	normal = normalize(cross(c - a, b - a));
#endif
	return normal;
}

#define GEOMETRY_WATER 1

inline __device__ void OceanShader(Float3& rayDir, Float3& beta, float clockTime)
{
	const float waterDepth = 1;

	// define bounding geometry
	const Float4 waterFloor = Float4(0.0, 1.0, 0.0, waterDepth);
	const Float4 waterCeiling = Float4(0.0, 1.0, 0.0, 0);

	// define ray
	Float3 orig = Float3(0.0, 2, 0.0);
	Ray ray(orig, rayDir);

	// trace bounding geometry
	float tError;

	float waterCeilingT = RayPlaneIntersect(waterCeiling, ray, tError);
	if (waterCeilingT > 1e4f) { return; }
	Float3 waterCeilingHitPos = GetRayPlaneIntersectPoint(waterCeiling, ray, waterCeilingT, tError);

#if GEOMETRY_WATER

	float waterFloorT = RayPlaneIntersect(waterFloor, ray, tError);
	if (waterFloorT > 1e4f) { return; }
	Float3 waterFloorHitPos = GetRayPlaneIntersectPoint(waterFloor, ray, waterFloorT, tError);

	// ray march from ceiling to floor
	float dist;
	Float3 pos;
	RayMarchWater(dist, pos, orig, waterCeilingHitPos, waterFloorHitPos, clockTime);

#else

	Float3 pos = waterCeilingHitPos;
	float dist = distance(pos, orig);

#endif

	// normal blending
	float distFallOff = 1.0 / (dist * dist * 1e-4 + 1.0);
	Float3 normal = GetWaterNormal(pos.xz(), waterDepth, clockTime, distFallOff);
	normal = mixf(Float3(0.0, 1.0, 0.0), normal, distFallOff);

	// fresnel
	normal = dot(normal, rayDir) > 0 ? Float3(0.0, 1.0, 0.0) : normal;
	float cosTheta = -dot(normal, rayDir);
	Float3 fresnel = Float3(0.04 + 0.96 * powf(1.0 - cosTheta, 5.0));

	// ray dir
	rayDir = normalize(reflect3f(rayDir, normal));

	// beta
	beta = fresnel;

#if GEOMETRY_WATER

	// add light wave color
	float height = min1f(powf(cosf((pos.y / waterDepth) * M_PI / 2), 16) * 10, 1.0);
	beta += height * (Float3(1.0) - fresnel) * Float3(0.8, 0.9, 0.6);
	beta = min3f(beta, Float3(1.0));

#endif
}
