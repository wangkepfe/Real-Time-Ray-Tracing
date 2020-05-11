#pragma once

#include <cuda_runtime.h>
#include "geometry.cuh"
#include "debug_util.cuh"

__device__ float RayleighPhaseFunc(float mu)
{
	return
			3. * (1. + mu*mu)
	/ //------------------------
				(16. * PI);
}


__device__ float HenyeyGreensteinPhaseFunc(float mu, float g)
{
	return
						(1. - g*g)
	/ //---------------------------------------------
		((4. * PI) * pow(1. + g*g - 2.*g*mu, 1.5));
}

__device__ Float3 GetEnvIncidentLight(const Float3& raydir, const Float3& sunDir)
{
	const float earthRadius = 6360e3; // (m)
	const float atmosphereRadius = 6420e3; // (m)

	const float sunPower = 20.0;

    const int numSamples = 16;
    const int numSamplesLight = 8;

    const float g = 0.76;

    // scattering coefficients at sea level (m)
    const Float3 betaR = Float3(5.5e-6, 13.0e-6, 22.4e-6); // Rayleigh
    const Float3 betaM = Float3(21e-6); // Mie

    // scale height (m) thickness of the atmosphere if its density were uniform
    const float hR = 7994.0; // Rayleigh
    const float hM = 1200.0; // Mie

    Ray eyeRay;
    eyeRay.orig = Float3(0, earthRadius + 2e3f, 0);
    eyeRay.dir = raydir;

    Sphere atmosphereSphere;
    atmosphereSphere.center = 0;
    atmosphereSphere.radius = atmosphereRadius;

    float errorSphereRayIntersect;
	float t = SphereRayIntersect(atmosphereSphere, eyeRay, errorSphereRayIntersect);

	if (t == RayMax)
	{
		return Float3(0.2);
	}

    float marchStep = t / float(numSamples + 1);

    float mu = dot(eyeRay.dir, sunDir);

    float phaseR = RayleighPhaseFunc(mu);
	float phaseM = HenyeyGreensteinPhaseFunc(mu, g);

    float opticalDepthR = 0.;
	float opticalDepthM = 0.;

	Float3 sumR = Float3(0);
	Float3 sumM = Float3(0);
	float marchPos = 0.;

    for (int i = 0; i < numSamples; i++)
    {
		Float3 s = eyeRay.orig + eyeRay.dir * (marchPos + 0.5 * marchStep);

		float height = s.length() - earthRadius;

		// integrate the height scale
		float hr = expf(-height / hR) * marchStep;
		float hm = expf(-height / hM) * marchStep;

		opticalDepthR += hr;
		opticalDepthM += hm;

		// gather the sunlight
        Ray lightRay;
        lightRay.orig = s;
        lightRay.dir = sunDir;

		float opticalDepthLightR = 0.;
		float opticalDepthLightM = 0.;

		bool overground = true;

		float t1 = SphereRayIntersect(atmosphereSphere, lightRay, errorSphereRayIntersect);

		if (t1 == RayMax)
		{
			continue;
		}

		float sunLightMarchPos = 0.;
		float sunLightMarchStep = t1 / float(numSamplesLight + 1);

		for (int i = 0; i < numSamplesLight; i++)
		{
			Float3 s1 = lightRay.orig + lightRay.dir * (sunLightMarchPos + 0.5 * sunLightMarchStep);
			float height1 = s1.length() - earthRadius;

			if (height1 < 0.) overground = false;

			opticalDepthLightR += expf(-height1 / hR) * sunLightMarchStep;
			opticalDepthLightM += expf(-height1 / hM) * sunLightMarchStep;

			sunLightMarchPos += sunLightMarchStep;
		}

		if (overground)
		{
			Float3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1 * (opticalDepthM + opticalDepthLightM);
			Float3 attenuation = exp3f(-tau);

			sumR += hr * attenuation;
			sumM += hm * attenuation;
		}

		marchPos += marchStep;
	}

	Float3 result = sunPower * (sumR * phaseR * betaR + sumM * phaseM * betaM);

	Float3 scat = Float3(1.0f) - clamp3f(opticalDepthM * 1e-5f, Float3(0), Float3(1));

	//result += scat;

	return result;
}

__device__ Float3 EnvLight(const Float3& raydir, const Float3& sunDir)
{
	Float3 sunOrMoonDir = sunDir;

	if (sunOrMoonDir.y > 0.2)
	{
		Float3 sunColor = GetEnvIncidentLight(raydir, sunOrMoonDir);
		return sunColor;
	}
	else if (sunOrMoonDir.y < 0.2 && sunOrMoonDir.y > -0.2)
	{
		Float3 sunColor = GetEnvIncidentLight(raydir, sunOrMoonDir);
		sunOrMoonDir.x *= -1;
		sunOrMoonDir.y *= -1;
		Float3 moonColor = GetEnvIncidentLight(raydir, sunOrMoonDir);
		return sunColor + moonColor * 0.05;
	}
	else
	{
		sunOrMoonDir.x *= -1;
		sunOrMoonDir.y *= -1;
		Float3 moonColor = GetEnvIncidentLight(raydir, sunOrMoonDir);
		return moonColor * 0.05;
	}
}