#pragma once

#include <cuda_runtime.h>
#include "geometry.cuh"
#include "debug_util.cuh"
#include "sampler.cuh"

//--------------------------------------------------------------------------
//Starfield
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// Return random noise in the range [0.0, 1.0], as a function of x.
inline __device__ float StarNoise2d(const Float2& v)
{
    float xhash = cosf( v.x * 37.0 );
    float yhash = cosf( v.y * 57.0 );
	float intPart;
	float val = modff( 415.92653 * ( xhash + yhash ) , &intPart);
	val = val < 0 ? val + 1 : val;
    return val;
}

// Convert Noise2d() into a "star field" by stomping everthing below fThreshhold to zero.
inline __device__ float NoisyStarField(const Float2& vSamplePos, float fThreshhold )
{
    float StarVal = StarNoise2d( vSamplePos );

    if ( StarVal >= fThreshhold )
        StarVal = powf( (StarVal - fThreshhold) / (1.0 - fThreshhold), 6.0 );
    else
        StarVal = 0.0;
    return StarVal;
}

// Stabilize NoisyStarField() by only sampling at integer values.
inline __device__ float StableStarField(const Float2& vSamplePos, float fThreshhold )
{
    // Linear interpolation between four samples.
    // Note: This approach has some visual artifacts.
    // There must be a better way to "anti alias" the star field.
	float intPart;

    float fractX = modff( vSamplePos.x , &intPart);
    float fractY = modff( vSamplePos.y , &intPart);

	fractX = fractX < 0 ? fractX + 1 : fractX;
	fractY = fractY < 0 ? fractY + 1 : fractY;

    Float2 floorSample = floor( vSamplePos );
    float v1 = NoisyStarField( floorSample, fThreshhold );
    float v2 = NoisyStarField( floorSample + Float2( 0.0, 1.0 ), fThreshhold );
    float v3 = NoisyStarField( floorSample + Float2( 1.0, 0.0 ), fThreshhold );
    float v4 = NoisyStarField( floorSample + Float2( 1.0, 1.0 ), fThreshhold );

    float StarVal =   v1 * ( 1.0 - fractX ) * ( 1.0 - fractY )
        			+ v2 * ( 1.0 - fractX ) * fractY
        			+ v3 * fractX * ( 1.0 - fractY )
        			+ v4 * fractX * fractY;
	return StarVal;
}

inline __device__ float RayleighPhaseFunc(float mu)
{
	return
			3. * (1. + mu*mu)
	/ //------------------------
				(16. * PI);
}


inline __device__ float HenyeyGreensteinPhaseFunc(float mu, float g)
{
	return
						(1. - g*g)
	/ //---------------------------------------------
		((4. * PI) * pow(1. + g*g - 2.*g*mu, 1.5));
}

inline __device__ Float3 GetEnvIncidentLight(const Float3& raydir, const Float3& sunDir, int numSamples, int numSamplesLight)
{
	const float earthRadius = 6360e3; // (m)
	const float atmosphereRadius = 6420e3; // (m)

	const float sunPower = 20.0;

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

	return result;
}

inline __device__ float GetWaves(Float2 pos, int iterations, float clockTime)
{
	pos *= 0.1;

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

#define WATER_NORMAL_QUALITY 20

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

#define GEOMETRY_WATER 0

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
	float height = min1f(powf(cosf((pos.y / waterDepth) * PI / 2), 16) * 10, 1.0);
	beta += height * (Float3(1.0) - fresnel) * Float3(0.8, 0.9, 0.6);
	beta = min3f(beta, Float3(1.0));

#endif
}

inline __device__ Float3 EnvLight(const Float3& raydir, const Float3& sunDir, float clockTime, bool isDiffuseRay)
{
	Float3 sunOrMoonDir = sunDir;

	Float3 rayDirOrRefl = raydir;
	Float3 beta = Float3(1.0);

	int numSkySample = 16;
	int numSkyLightSample = 8;

	if (rayDirOrRefl.y < 0.01 && isDiffuseRay == false) { OceanShader(rayDirOrRefl, beta, clockTime * 0.7); }

	if (rayDirOrRefl.y < 0) { rayDirOrRefl.y = -rayDirOrRefl.y; }

	if ((sunOrMoonDir.y > -0.05 && sunOrMoonDir.z >= 0) || (sunOrMoonDir.y > 0.05 && sunOrMoonDir.z < 0))
	{
		Float3 sunColor = GetEnvIncidentLight(rayDirOrRefl, sunOrMoonDir, numSkySample, numSkyLightSample);
		sunColor = sunColor * (powf(max1f(dot(rayDirOrRefl, sunOrMoonDir), 0), 500.0) * 1.0 + 1.0);
		return sunColor * beta;
	}
	else
	{
		sunOrMoonDir = -sunOrMoonDir;

		Float3 moonColor = GetEnvIncidentLight(rayDirOrRefl, sunOrMoonDir, numSkySample, numSkyLightSample);
		moonColor *= 0.01;
		moonColor += moonColor * powf(max1f(dot(rayDirOrRefl, sunOrMoonDir), 0), 55);
		if (dot(rayDirOrRefl, sunOrMoonDir) > 0.999)
			moonColor += Float3(0.9608, 0.9529, 0.8078) * 0.1;

		float starColor = 0;
#if 0
		if (isDiffuseRay == false)
		{
			Float2 uv = Float2((atan2f(rayDirOrRefl.x, rayDirOrRefl.z) + clockTime / 300) / TWO_PI, acosf(rayDirOrRefl.y) / PI) * 6000.0;
			starColor = StableStarField(uv, 0.995f) * 0.1;
		}
#endif
		return (moonColor + starColor) * beta;
	}
}

inline __device__ Float3 EnvLight2(const Float3& raydir, float clockTime, bool isDiffuseRay, TexObj skyTex)
{
	Float3 rayDirOrRefl = raydir;
	Float3 beta = Float3(1.0);

	if (rayDirOrRefl.y < 0.01 && isDiffuseRay == false) { OceanShader(rayDirOrRefl, beta, clockTime * 0.7); }

	float u = atan2f(-rayDirOrRefl.z, -rayDirOrRefl.x) / TWO_PI + 0.5f;
	float v = abs(rayDirOrRefl.y);

	float4 texRead = tex2D<float4>(skyTex, u, v);
	Float3 color = Float3(texRead.x , texRead.y, texRead.z);

	return color * beta;
}

__global__ void Sky(SurfObj skyBuffer, float* skyCdf, Int2 size, Float3 sunDir)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = ((float)x + 0.5f) / size.x;
	float v = ((float)y + 0.5f) / size.y;

	// hemisphere projection
	float z = v;
	float r = sqrtf(1 - v * v);
	float phi = TWO_PI * u;

	// ray dir
	Float3 rayDir(r * cosf(phi), z, r * sinf(phi));

	// get sky color
	const int numSkySample = 16;
	const int numSkyLightSample = 8;
	Float3 sunOrMoonDir = sunDir;
	Float3 color;
	if ((sunOrMoonDir.y > -0.05 && sunOrMoonDir.z >= 0) || (sunOrMoonDir.y > 0.05 && sunOrMoonDir.z < 0))
	{
		Float3 sunColor = GetEnvIncidentLight(rayDir, sunOrMoonDir, numSkySample, numSkyLightSample);
		sunColor = sunColor * (powf(max1f(dot(rayDir, sunOrMoonDir), 0), 500.0) * 1.0 + 1.0);
		color = sunColor;
	}
	else
	{
		sunOrMoonDir = -sunOrMoonDir;
		Float3 moonColor = GetEnvIncidentLight(rayDir, sunOrMoonDir, numSkySample, numSkyLightSample) * 0.01;
		moonColor += moonColor * powf(max1f(dot(rayDir, sunOrMoonDir), 0), 55);
		color = moonColor;
	}

	// store
	Store2D(Float4(color, 0), skyBuffer, Int2(x, y));

	// sky cdf
	int i = size.x * y + x;
	skyCdf[i] = color.max();
}

__global__ void Scan(float *data, int n)
{
	// allocated on invocation
	extern __shared__ float temp[];

	int i = threadIdx.x;
	int offset = 1;

	// load input into shared memory
	temp[2 * i] = data[2 * i];
	temp[2 * i + 1] = data[2 * i + 1];

	// save orig value
	float orig0 = temp[2 * i];
	float orig1 = temp[2 * i + 1];

	// build sum in place up the tree
	for (int d = n>>1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (i < d)
		{
			int ai = offset * (2 * i + 1) - 1;
			int bi = offset * (2 * i + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// clear the last element
	if (i == 0)
	{
		temp[n - 1] = 0;
	}

	// traverse down tree & build scan
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();
		if (i < d)
		{
			int ai = offset * (2 * i + 1) - 1;
			int bi = offset * (2 * i + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	// write results to device memory
	data[2 * i] = temp[2 * i] + orig0;
	data[2 * i + 1] = temp[2 * i + 1] + orig1;

	//printf("data[%d] = %f\n", 2 * i, data[2 * i]);
	//("data[%d] = %f\n", 2 * i + 1, data[2 * i + 1]);
}
