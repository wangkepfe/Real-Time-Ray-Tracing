#pragma once

#include <cuda_runtime.h>
#include "linear_math.h"

__device__  void LocalizeSample(
	const Float3& n,
	Float3& u,
	Float3& v)
{
	Float3 w;

	if (abs(n.y) < 1.0 - 1e-5)
		w = Float3(0, 1, 0);
	else
		w = Float3(1, 0, 0);

	u = normalize(cross(n, w));
	v = normalize(cross(n, u));
}

__device__  Float2 ConcentricSampleDisk(Float2 u) {
	// Map uniform random numbers to [-1, 1]
	Float2 uOffset = 2.0 * u - 1.0;

	// Handle degeneracy at the origin
	if (abs(uOffset.x) < 1e-10 && abs(uOffset.y) < 1e-10) {
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

__device__  Float3 CosineSampleHemisphere(Float2 u)
{
	Float2 d = ConcentricSampleDisk(u);
	float z = sqrtf(max1f(0.0f, 1.0f - d.x * d.x - d.y * d.y));
	return Float3(d.x, z, d.y);
}

__device__  Float3 CosineSampleHemisphere(Float2 u, const Float3& x, const Float3& y, const Float3& z)
{
	Float2 d = ConcentricSampleDisk(u);
	float dz = sqrtf(max1f(0.0f, 1.0f - d.x * d.x - d.y * d.y));
	return x * d.x + y * d.y + z * dz;
}

__device__  void LambertianSample(
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

__device__ inline Float3 LambertianBsdf(const Float3& wo, const Float3& n, const Float3& albedo) { return max1f(dot(wo, n), 0) * albedo / PI; }
__device__ inline float  LambertianPdf(const Float3& wo, const Float3& n) { return max1f(dot(wo, n), 0) / PI; }
__device__ inline Float3 LambertianBsdfOverPdf(const Float3& albedo) { return albedo; }

__device__ float FresnelDialetric(float etaI, float etaT, float cosThetaI, float cosThetaT)
{
	float R1 = etaT * cosThetaI;
	float R2 = etaI * cosThetaT;
	float R3 = etaI * cosThetaI;
	float R4 = etaT * cosThetaT;

	float Rparl = (R1 - R2) / (R1 + R2);
	float Rperp = (R3 - R4) / (R3 + R4);

	return (Rparl * Rparl + Rperp * Rperp) / 2.0;
}

__device__ inline float pow5(float e) {
	float e2 = e * e;
	return e2 * e2 * e;
}

__device__ inline Float3 FresnelShlick(const Float3& F0, float cosTheta) {
	return F0 + (Float3(1.0f) - F0) * pow5(1.0f - cosTheta);
}

__device__ inline float FresnelShlick(float F0, float cosTheta) {
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

__device__ void MacrofacetReflection(
	float r1,
	float r2,

	const Float3& raydir,
	Float3& nextdir,
	const Float3& normal,

	Float3& beta,
	const Float3& F0,

	float alpha)
{
	// pre-calculate
	float alpha2 = alpha * alpha;

	// sample normal
	Float3 sampledNormalLocal;
	float cosTheta = 1.0f / sqrt(1.0f + alpha2 * r1 / (1.0f - r1));
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
	float phi = TWO_PI * r2;
	sampledNormalLocal = Float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

	// local to world space
	Float3 t, b;
	LocalizeSample(normal, t, b);
	Float3 sampledNormal = sampledNormalLocal.x * t + sampledNormalLocal.z * b + sampledNormalLocal.y * normal;
	sampledNormal.normalize();

	// reflect
	nextdir = raydir - sampledNormal * dot(sampledNormal, raydir) * 2.0f;
	nextdir.normalize();

	// Fresnel
	float cosThetaWoWh = max(0.01f, abs(dot(sampledNormal, nextdir)));
	Float3 F = FresnelShlick(F0, cosThetaWoWh);

	// Smith's Mask-shadowing function G
	float cosThetaWo = abs(dot(nextdir, normal));
	float cosThetaWi = max(0.01f, abs(dot(raydir, normal)));
	float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
	float G = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

	// color
	float cosThetaWh = max(0.01f, dot(sampledNormal, normal));
	beta = minf3f(1.0f, F * G * cosThetaWoWh / cosThetaWi / cosThetaWh);
}

__device__ Float3 UniformSampleCone(const Float2 &u, float cosThetaMax, const Float3 &x, const Float3 &y, const Float3 &z)
{
    float cosTheta = ((float)1 - u[0]) + u[0] * cosThetaMax;
    float sinTheta = sqrtf((float)1 - cosTheta * cosTheta);
    float phi = u[1] * 2 * PI;
    return cosf(phi) * sinTheta * x + sinf(phi) * sinTheta * y + cosTheta * z;
}

__device__ float UniformConePdf(float cosThetaMax)
{
	return 1 / (2 * PI * (1.0 - cosThetaMax));
}

__device__ Float3 UniformSampleSphere(const Float2 &u)
{
    float z = 1 - 2 * u[0];
    float r = sqrtf(max1f((float)0, (float)1 - z * z));
    float phi = 2 * PI * u[1];
    return Float3(r * cosf(phi), r * sinf(phi), z);
}

__device__ constexpr float UniformSpherePdf() { return 1.0f / (4.0f * PI); }

__device__ Float3 UniformSampleHemisphere(const Float2 &u)
{
    float z = u[0];
    float r = sqrtf(max1f((float)0, (float)1. - z * z));
    float phi = 2 * PI * u[1];
    return Float3(r * cosf(phi), r * sinf(phi), z);
}

__device__ constexpr float UniformHemispherePdf() { return 1.0f / (2.0f * PI); }