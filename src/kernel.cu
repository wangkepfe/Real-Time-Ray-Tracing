#include "kernel.cuh"
#include "debug_util.cuh"
#include "geometry.cuh"
#include "bsdf.cuh"
#include "morton.cuh"
#include "sampler.cuh"
#include "denoise.cuh"
#include "sky.cuh"
#include "postprocessing.cuh"
#include "raygen.cuh"
#include "traverse.cuh"
#include "hash.cuh"

__device__ inline void LightShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial)
{
	// check for termination and hit light
	if (rayState.terminated == true || rayState.hitLight == false) { return; }

	// ray is terminated
	rayState.terminated = true;

	// Different light source type
	if (rayState.matType == MAT_SKY)
	{
		// env light
		Float3 envLightColor = EnvLight(rayState.dir, cbo.sunDir);
		rayState.L += envLightColor * rayState.beta;
	}
	else if (rayState.matType == EMISSIVE)
	{
		// local light
		SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];
		Float3 L0 = mat.albedo;
		rayState.L += L0 * rayState.beta;
	}
}

__device__ inline void GlossyShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial)
{
	// check for termination and hit light
	if (rayState.terminated == true || rayState.hitLight == true || rayState.isDiffuse == true) { return; }

	if (rayState.matType == PERFECT_REFLECTION)
	{
		// mirror
		rayState.dir = normalize(rayState.dir - rayState.normal * dot(rayState.dir, rayState.normal) * 2.0);
		rayState.orig = rayState.pos + rayState.offset * rayState.normal;
	}
	else if (rayState.matType == PERFECT_FRESNEL_REFLECTION_REFRACTION)
	{
		// glass
		Float3 nextRayDir;
		float rayOffset = rayState.offset;
		PerfectReflectionRefraction(1.0, 1.33, rayState.isRayIntoSurface, rayState.normal, rayState.normalDotRayDir, rd(&rayState.rdState[2]), rayState.dir, nextRayDir, rayOffset);
		rayState.dir = nextRayDir;
		rayState.orig = rayState.pos + rayOffset * rayState.normal;
	}
}

__device__ inline void DiffuseShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial)
{
	// check for termination and hit light
	if (rayState.terminated == true || rayState.hitLight == true || rayState.isDiffuse == false) { return; }

	// get mat
	SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];

	if (rayState.matType == LAMBERTIAN_DIFFUSE)
	{
		// surface sample
		LambertianSample(rd2(&rayState.rdState[0], &rayState.rdState[1]), rayState.dir, rayState.normal);
		Float3 surfaceSamplebsdfOverPdf = LambertianBsdfOverPdf(mat.albedo);
		float surfaceSamplePdf = LambertianPdf(rayState.dir, rayState.normal);

		// light sample
		/// @todo

		// update ray
		rayState.orig = rayState.pos + rayState.offset * rayState.normal;

		// update beta
		rayState.beta *= surfaceSamplebsdfOverPdf;
	}
	else if (rayState.matType == MICROFACET_REFLECTION)
	{
		const Float3 F0(0.56, 0.57, 0.58);
		const float alpha = 0.05;

		Float3 nextRayDir;
		MacrofacetReflection(rd(&rayState.rdState[0]), rd(&rayState.rdState[1]), rayState.dir, nextRayDir, rayState.normal, rayState.beta, F0, alpha);
		rayState.dir = nextRayDir;

		// light sample
		/// @todo

		// update ray
		rayState.orig = rayState.pos + rayState.offset * rayState.normal;
	}
}

__device__ inline void UpdateMaterial(RayState& rayState, SceneMaterial sceneMaterial)
{
	if (rayState.objectIdx == 998)
	{
		rayState.matType = PERFECT_REFLECTION;
	}
	else
	{
		rayState.matId = sceneMaterial.materialsIdx[rayState.objectIdx];
		SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];
		rayState.matType = (rayState.hit == false) ? MAT_SKY : mat.type;
	}
	rayState.hitLight = (rayState.matType == MAT_SKY) || (rayState.matType == EMISSIVE);
	rayState.isDiffuse = (rayState.matType == LAMBERTIAN_DIFFUSE) || (rayState.matType == MICROFACET_REFLECTION);
}

__global__ void PathTrace(ConstBuffer cbo, SceneGeometry sceneGeometry, SceneMaterial sceneMaterial, RandInitVec* randInitVec, SurfObj colorBuffer, SurfObj normalBuffer, SurfObj positionBuffer)
{
	// index
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;

	// init ray state
	RayState rayState;
	rayState.i          = i;
	rayState.L          = 0.0;
	rayState.beta       = 1.0;
	rayState.idx        = idx;
	rayState.bounce     = 0;
	rayState.terminated = false;
	rayState.distance   = 0;

	// init rand state
	const uint seed = (idx.x << 16) ^ (idx.y);
	for (int k = 0; k < 3; ++k) { curand_init(randInitVec[k], CURAND_2POW32 * 0.5f, WangHash(seed) + cbo.frameNum, &rayState.rdState[k]); }

	// generate ray
	GenerateRay(rayState.orig, rayState.dir, cbo.camera, idx, rd2(&rayState.rdState[0], &rayState.rdState[1]));

	// scene traverse
	rayState.hit = RaySceneIntersect(Ray(rayState.orig, rayState.dir), sceneGeometry, rayState.pos, rayState.normal, rayState.objectIdx, rayState.offset, rayState.distance, rayState.isRayIntoSurface, rayState.normalDotRayDir);

	// store normal, pos
	Store2D(Float4(rayState.normal, 1.0), normalBuffer, idx);
	Store2D(Float4(rayState.pos, 1.0), positionBuffer, idx);

	// update mat id
	UpdateMaterial(rayState, sceneMaterial);

	for (int bounce = 0; bounce < 4; ++bounce)
	{
		LightShader(cbo, rayState, sceneMaterial);
		GlossyShader(cbo, rayState, sceneMaterial);
		DiffuseShader(cbo, rayState, sceneMaterial);

		if (bounce != 3)
		{
			rayState.hit = RaySceneIntersect(Ray(rayState.orig, rayState.dir), sceneGeometry, rayState.pos, rayState.normal, rayState.objectIdx, rayState.offset, rayState.distance, rayState.isRayIntoSurface, rayState.normalDotRayDir);
			UpdateMaterial(rayState, sceneMaterial);
		}
	}

	// write to buffer
	Store2D(Float4(rayState.L, 1.0), colorBuffer, idx);
}

void RayTracer::draw(SurfObj* renderTarget)
{
	// ---------------- frame update ------------------
	timer.update();
	float clockTime   = timer.getTime();

	const Float3 axis = normalize(Float3(0.0, 0.0, 1.0));
	const float angle = fmodf(clockTime * TWO_PI / 10, TWO_PI);
	Float3 sunDir     = rotate3f(axis, angle, Float3(0.0, 1.0, 2.5)).normalized();

	cbo.sunDir        = sunDir;
	cbo.frameNum      = cbo.frameNum + 1;

	// ----- scene -----
	// sphere
	Float3 spherePos = Float3(sinf(clockTime * TWO_PI / 5) * 0.01f, terrain.getHeightAt(0.0f) + 0.005f, 0.0f);
	cameraFocusPos        = Float3(0.0f, terrain.getHeightAt(0.0f) + 0.005f, 0.0f);
	spheres[0]            = Sphere(spherePos, 0.005f);
	sphereLights[0]           = spheres[0];

	GpuErrorCheck(cudaMemcpyAsync(d_spheres          , spheres      , numSpheres *      sizeof(Float4)         , cudaMemcpyHostToDevice, streams[2]));
	GpuErrorCheck(cudaMemcpyAsync(d_sphereLights     , sphereLights , numSphereLights * sizeof(Float4)         , cudaMemcpyHostToDevice, streams[2]));

	d_sceneGeometry.numSpheres      = numSpheres;
	d_sceneGeometry.spheres         = d_spheres;

	d_sceneMaterial.numSphereLights = numSphereLights;
	d_sceneMaterial.sphereLights    = d_sphereLights;

	// camera
	// Camera& camera = cbo.camera;
	// Float3 cameraLookAtPoint = cameraFocusPos + Float3(0.0f, 0.01f, 0.0f);
	// camera.pos               = cameraFocusPos + rotate3f(Float3(0, 1, 0), fmodf(clockTime * TWO_PI / 60, TWO_PI), Float3(0.0f, 0.0f, -0.1f)) + Float3(0, abs(sinf(clockTime * TWO_PI / 60)) * 0.05f, 0);
	// camera.dir               = normalize(cameraLookAtPoint - camera.pos.xyz);
	// camera.left              = cross(Float3(0, 1, 0), camera.dir.xyz);

	Int2 bufferDim(renderWidth, renderHeight);
	Int2 outputDim(screenWidth, screenHeight);

	// ------------------ Ray Gen -------------------
	PathTrace<<<gridDim, blockDim, 0, streams[0]>>>(cbo, d_sceneGeometry, d_sceneMaterial, d_randInitVec, colorBufferA, normalBuffer, positionBuffer);

	// ---------------- post processing ----------------
	ToneMapping<<<gridDim, blockDim, 0, streams[0]>>>(/*io*/colorBufferA , bufferDim , /*exposure*/1.0);
	Denoise    <<<gridDim, blockDim, 0, streams[0]>>>(/*io*/colorBufferA , /*in*/normalBuffer , /*in*/positionBuffer, bufferDim, cbDenoise);

	if (cbo.frameNum == 1) { BufferCopy<<<gridDim, blockDim, 0, streams[0]>>>(/*out*/colorBufferB , /*in*/colorBufferA , bufferDim); }
	else                   { TAA       <<<gridDim, blockDim, 0, streams[0]>>>(/*io*/colorBufferB  , /*in*/colorBufferA , bufferDim); }

	FilterScale<<<scaleGridDim, scaleBlockDim, 0, streams[0]>>>(/*out*/renderTarget, /*in*/colorBufferB, outputDim, bufferDim);
}