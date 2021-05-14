#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"

#define SAFE_COSINE_EPSI 1e-5f

__device__ __forceinline__ void LocalizeSample(
	const Float3& n,
	Float3& u,
	Float3& v)
{
	Float3 w;

	if (abs(n.y) < 1.0f - 1e-5f)
		w = Float3(0, 1, 0);
	else
		w = Float3(1, 0, 0);

	u = cross(n, w);
	v = cross(n, u);
}

__device__ __forceinline__ Float2 ConcentricSampleDisk(Float2 u) {
	// Map uniform random numbers to [-1, 1]
	Float2 uOffset = 2.0 * u - 1.0;

	// Handle degeneracy at the origin
	if (abs(uOffset.x) < 1e-10f && abs(uOffset.y) < 1e-10f) {
		return Float2(0, 0);
	}

	// Apply concentric mapping to point
	float theta;
	float r;

	if (abs(uOffset.x) > abs(uOffset.y))
	{
		r = uOffset.x;
		theta = PI_OVER_4 * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = PI_OVER_2 - PI_OVER_4 * (uOffset.x / uOffset.y);
	}

	return r * Float2(cosf(theta), sinf(theta));
}

__device__ __forceinline__ Float2 UniformSampleDisk(Float2 u) {
    float r = sqrtf(u[0]);
    float theta = TWO_PI * u[1];
    return Float2(r * cosf(theta), r * sinf(theta));
}

#define DISK_SAMPLE_CONCENTRIC 1
#define DISK_SAMPLE_UNIFORM !DISK_SAMPLE_CONCENTRIC

__device__ __forceinline__ Float3 CosineSampleHemisphere(Float2 u)
{
#if DISK_SAMPLE_CONCENTRIC
	Float2 d = ConcentricSampleDisk(u);
#elif DISK_SAMPLE_UNIFORM
	Float2 d = UniformSampleDisk(u);
#endif

	float z = sqrtf(max1f(0.0f, 1.0f - d.x * d.x - d.y * d.y));
	return Float3(d.x, z, d.y);
}

__device__ __forceinline__ Float3 CosineSampleHemisphere(Float2 u, const Float3& x, const Float3& y, const Float3& z)
{
#if DISK_SAMPLE_CONCENTRIC
	Float2 d = ConcentricSampleDisk(u);
#elif DISK_SAMPLE_UNIFORM
	Float2 d = UniformSampleDisk(u);
#endif

	float dz = sqrtf(max1f(0.0f, 1.0f - d.x * d.x - d.y * d.y));
	return x * d.x + y * d.y + z * dz;
}

__device__ __forceinline__ void LambertianSample(
	Float2         randNum,
	Float3&        wo,
	const Float3&  n)
{
	Float3 s = CosineSampleHemisphere(randNum);

	Float3 u, v;
	LocalizeSample(n, u, v);

	wo = s.x * u + s.z * v + s.y * n;
	wo.normalize();
}

__device__ __forceinline__ Float3 LambertianBsdf(const Float3& albedo) { return albedo / M_PI; }
__device__ __forceinline__ float  LambertianPdf(const Float3& wi, const Float3& n) { return max(dot(wi, n), SAFE_COSINE_EPSI) / M_PI; }
__device__ __forceinline__ Float3 LambertianBsdfOverPdf(const Float3& albedo) { return albedo; }

__device__ __forceinline__ float FresnelDialetric(float etaI, float etaT, float cosThetaI, float cosThetaT)
{
	float R1 = etaT * cosThetaI;
	float R2 = etaI * cosThetaT;
	float R3 = etaI * cosThetaI;
	float R4 = etaT * cosThetaT;

	float Rparl = (R1 - R2) / (R1 + R2);
	float Rperp = (R3 - R4) / (R3 + R4);

	return (Rparl * Rparl + Rperp * Rperp) / 2.0;
}

__device__ __forceinline__ float pow5(float e) {
	float e2 = e * e;
	return e2 * e2 * e;
}

__device__ __forceinline__ Float3 FresnelShlick(const Float3& F0, float cosTheta) {
	return F0 + (Float3(1.0f) - F0) * pow5(1.0f - cosTheta);
}

__device__ __forceinline__ float FresnelShlick(float F0, float cosTheta) {
	return F0 + (1.0f - F0) * pow5(1.0f - cosTheta);
}

__device__ void PerfectReflectionRefraction(float etaI, float etaT, bool isRayIntoSurface, Float3 normal, float normalDotRayDir, float uniformRandNum, Float3 rayDir, Float3& nextRayDir, float& rayoffset)
{
	// eta
	if (isRayIntoSurface == false) swap(etaI, etaT);
	const float eta = etaI/etaT;

	// trigonometry
	float cosThetaI  = -normalDotRayDir;
	float sin2ThetaI = max1f(0, 1.0 - cosThetaI * cosThetaI);
	float sin2ThetaT = eta * eta * sin2ThetaI;
	float cosThetaT = sqrt(max1f(0, 1.0 - sin2ThetaT));

	// total internal reflection
	if (sin2ThetaT >= 1.0)
	{
		nextRayDir = rayDir - normal * normalDotRayDir * 2.0;
	}
	else
	{
		// Fresnel for dialectric
		float fresnel = FresnelDialetric(etaI, etaT, cosThetaI, cosThetaT);

		// reflection or transmittance
		if (uniformRandNum < fresnel)
		{
			nextRayDir = rayDir - normal * normalDotRayDir * 2.0;
		}
		else
		{
			nextRayDir = eta * rayDir + (eta * cosThetaI - cosThetaT) * normal;
			rayoffset *= -1.0;
		}
	}

	nextRayDir.normalize();
}

__forceinline__ __device__ void MacrofacetReflectionSample(
	Float2 r,
	Float2 r2,

	const Float3& raydir,
	Float3& nextdir,
	const Float3& normal,
	const Float3& surfaceNormal,

	Float3& brdfOverPdf,
	Float3& brdf,
	float& pdf,

	const Float3& F0,
	Float3& albedo,

	float alpha)
{
	// pre-calculate
	float alpha2 = alpha * alpha;

	// sample normal
	Float3 sampledNormalLocal;
	float cosTheta = 1.0f / sqrt(1.0f + alpha2 * r[0] / (1.0f - r[0]));
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
	float phi = TWO_PI * r[1];
	sampledNormalLocal = Float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

	// local to world space
	Float3 t, b;
	LocalizeSample(normal, t, b);
	Float3 sampledNormal = sampledNormalLocal.x * t + sampledNormalLocal.z * b + sampledNormalLocal.y * normal;
	sampledNormal.normalize();

	// reflect
	nextdir = normalize(reflect3f(raydir, sampledNormal));

	if (dot(nextdir, surfaceNormal) < 0)
	{
		cosTheta = 1.0f / sqrt(1.0f + alpha2 * r2[0] / (1.0f - r2[0]));
		sinTheta = sqrt(1.0f - cosTheta * cosTheta);
		phi = TWO_PI * r2[1];
		sampledNormalLocal = Float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

		// local to world space
		Float3 t, b;
		LocalizeSample(normal, t, b);
		Float3 sampledNormal = sampledNormalLocal.x * t + sampledNormalLocal.z * b + sampledNormalLocal.y * normal;
		sampledNormal.normalize();

		// reflect
		nextdir = normalize(reflect3f(raydir, sampledNormal));

		if (dot(nextdir, surfaceNormal) < 0)
		{
			nextdir = normalize(reflect3f(raydir, normal));
		}
	}

	Float3 wi = nextdir;
	Float3 wo = -raydir;
	Float3 wh = sampledNormal;
	Float3 wn = normal;

	// Fresnel
	float cosThetaWoWh = max(SAFE_COSINE_EPSI, dot(wh, wo));
	Float3 F = FresnelShlick(F0, cosThetaWoWh);

	// Smith's Mask-shadowing function G
	float cosThetaWo = max(SAFE_COSINE_EPSI, dot(wo, wn));
	float cosThetaWi = max(SAFE_COSINE_EPSI, dot(wi, wn));
	float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
	float G = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

	// Trowbridge Reitz Distribution D
	float cosThetaWh = max(SAFE_COSINE_EPSI, dot(wh, wn));
	float cosTheta2Wh = cosThetaWh * cosThetaWh;
	float tanTheta2Wh = (1.0f - cosTheta2Wh) / cosTheta2Wh;
	float e = tanTheta2Wh / alpha2 + 1.0f;
	float D = 1.0f / (M_PI * (alpha2 * cosTheta2Wh * cosTheta2Wh) * (e * e));

	// brdf
	brdf = (albedo * F) * (D * G) / (4.0f * cosThetaWo * cosThetaWi);

	// pdf
	pdf = (D * cosThetaWh) / (4.0f * cosThetaWoWh);

	// beta
	brdfOverPdf = (albedo * F) * (G * cosThetaWoWh) / (cosThetaWh * cosThetaWo); // brdf / pdf * cosThetaWi
}

__forceinline__ __device__ void MacrofacetReflection(Float3& brdfOverPdf, Float3& brdf, float& pdf,const Float3& wn, Float3 wo, Float3 wi, const Float3& F0, Float3& albedo, float alpha)
{
	float alpha2 = alpha * alpha;
	if (dot(wo, wn) < 0) wo = -wo;
	if (dot(wi, wn) < 0) wi = -wi;
	Float3 wh = normalize(wi + wo);

	// Fresnel
	float cosThetaWoWh = max(SAFE_COSINE_EPSI, dot(wh, wo));
	Float3 F           = FresnelShlick(F0, cosThetaWoWh);

	// Smith's Mask-shadowing function G
	float cosThetaWo = max(SAFE_COSINE_EPSI, dot(wo, wn));
	float cosThetaWi = max(SAFE_COSINE_EPSI, dot(wi, wn));
	float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
	float G          = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

	// Trowbridge Reitz Distribution D
	float cosThetaWh  = max(SAFE_COSINE_EPSI, dot(wh, wn));
	float cosTheta2Wh = cosThetaWh * cosThetaWh;
	float tanTheta2Wh = (1.0f - cosTheta2Wh) / cosTheta2Wh;
	float e           = tanTheta2Wh / alpha2 + 1.0f;
	float D           = 1.0f / (M_PI * (alpha2 * cosTheta2Wh * cosTheta2Wh) * (e * e));

	// brdf
	brdf = (albedo * F) * (D * G) / (4.0f * cosThetaWo * cosThetaWi);

	// pdf
	pdf = (D * cosThetaWh) / (4.0f * cosThetaWoWh);

	// beta
	brdfOverPdf = (albedo * F) * (G * cosThetaWoWh) / (cosThetaWh * cosThetaWo); // brdf / pdf * cosThetaWi
}

__forceinline__ __device__ void MacrofacetReflection2(
	Float3& brdfOverPdf, Float3& brdf, float& pdf,
	const Float3& F0, const Float3& albedo, float alpha,
	float cosThetaWoWh, float cosThetaWo, float cosThetaWi, float cosThetaWh)
{
	float alpha2 = alpha * alpha;

	// Fresnel
	Float3 F           = FresnelShlick(F0, cosThetaWoWh);

	// Smith's Mask-shadowing function G
	float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
	float G          = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

	// Trowbridge Reitz Distribution D
	float cosTheta2Wh = cosThetaWh * cosThetaWh;
	float tanTheta2Wh = (1.0f - cosTheta2Wh) / cosTheta2Wh;
	float e           = tanTheta2Wh / alpha2 + 1.0f;
	float D           = 1.0f / (M_PI * (alpha2 * cosTheta2Wh * cosTheta2Wh) * (e * e));

	// brdf
	brdf = (albedo * F) * (D * G) / (4.0f * cosThetaWo * cosThetaWi);

	// pdf
	pdf = (D * cosThetaWh) / (4.0f * cosThetaWoWh);

	// beta
	brdfOverPdf = (albedo * F) * (G * cosThetaWoWh) / (cosThetaWh * cosThetaWo); // brdf / pdf * cosThetaWi
}

__device__ __forceinline__ Float3 UniformSampleCone(const Float2 &u, float cosThetaMax, const Float3 &x, const Float3 &y, const Float3 &z)
{
    float cosTheta = (1.0f - u[0]) + u[0] * cosThetaMax;
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float phi = u[1] * TWO_PI;
    return cosf(phi) * sinTheta * x + sinf(phi) * sinTheta * y + cosTheta * z;
}

__device__ __forceinline__ float UniformConePdf(float cosThetaMax)
{
	return 1 / (2 * M_PI * (1.0 - cosThetaMax));
}

__device__ __forceinline__ Float3 UniformSampleSphere(const Float2 &u)
{
    float z = 1 - 2 * u[0];
    float r = sqrtf(max1f((float)0, (float)1 - z * z));
    float phi = 2 * M_PI * u[1];
    return Float3(r * cosf(phi), r * sinf(phi), z);
}

__device__ __forceinline__ constexpr float UniformSpherePdf() { return 1.0f / (4.0f * M_PI); }

__device__ __forceinline__ Float3 UniformSampleHemisphere(const Float2 &u)
{
    float z = u[0];
    float r = sqrtf(max1f((float)0, (float)1. - z * z));
    float phi = 2 * M_PI * u[1];
    return Float3(r * cosf(phi), r * sinf(phi), z);
}

__device__ __forceinline__ constexpr float UniformHemispherePdf() { return 1.0f / (2.0f * M_PI); }

__device__ __forceinline__ constexpr float PowerHeuristic(float f, float g) { return (f * f) / (f * f + g * g); }

__device__ __forceinline__ void CalculateCosines(
	const Float3& raydir, const Float3& nextdir, const Float3& normal,
	float& cosThetaWoWh, float& cosThetaWo, float& cosThetaWi, float& cosThetaWh)
{
	Float3 wi = nextdir;
	Float3 wo = -raydir;
	Float3 wn = normal;
	Float3 wh = normalize(wi + wo);
	cosThetaWoWh = max(SAFE_COSINE_EPSI, dot(wh, wo));
	cosThetaWo = max(SAFE_COSINE_EPSI, dot(wo, wn));
	cosThetaWi = max(SAFE_COSINE_EPSI, dot(wi, wn));
	cosThetaWh  = max(SAFE_COSINE_EPSI, dot(wh, wn));
}