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

__global__ void RayGen(ConstBuffer cbo, RayState* rayState, RandInitVec* randInitVec, SceneGeometry sceneGeometry, SurfObj normalBuffer, SurfObj positionBuffer, IndexBuffers indexBuffers)
{
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;
	RayState ray;
	ray.L          = 0.0;
	ray.beta       = 1.0;
	ray.idx        = idx;
	ray.bounce     = 0;
	ray.terminated = false;
	const uint seed = (idx.x << 16) ^ (idx.y);
	for (int k = 0; k < 3; ++k) { curand_init(randInitVec[k], CURAND_2POW32 * 0.5f, WangHash(seed) + cbo.frameNum, &ray.rdState[k]); }
	ray.ray = GenerateRay(cbo.camera, idx, rd2(&ray.rdState[0], &ray.rdState[1]));
	ray.hit = RaySceneIntersect(ray.ray, sceneGeometry, ray.pos, ray.normal, ray.objectIdx, ray.offset);
	Store2D(Float4(ray.normal, 1.0), normalBuffer, idx);
	Store2D(Float4(ray.pos, 1.0), positionBuffer, idx);
	rayState[i] = ray;

	uint* hitBuffer = indexBuffers.hitBufferA;
	uint* missBuffer = indexBuffers.missBufferA;
	uint* hitBufferTop = indexBuffers.hitBufferTopA;
	uint* missBufferTop = indexBuffers.missBufferTopA;
	if (ray.hit)
	{
		uint oldVal = atomicInc(hitBufferTop, cbo.bufferSize);
		hitBuffer[oldVal] = i;
	}
	else
	{
		uint oldVal = atomicInc(missBufferTop, cbo.bufferSize);
		missBuffer[oldVal] = i;
	}
}

__device__ inline void SkyShader(ConstBuffer& cbo, RayState* rayStates, SurfObj& colorBuffer, uint& i, Int2& idx)
{
	RayState& rayState = rayStates[i];
	if (rayState.terminated == true || rayState.hit == true) { return; }
	rayState.terminated = true;
	Float3 envLightColor = EnvLight(rayState.ray.dir, cbo.sunDir);
	Float3 outColor = rayState.L;
	outColor += envLightColor * rayState.beta;
	Store2D(Float4(outColor, 1.0), colorBuffer, idx);
}

__device__ inline void SurfaceShader(ConstBuffer& cbo, RayState* rayStates, SceneMaterial& sceneMaterial, SurfObj& colorBuffer, uint& i, Int2& idx, bool& shouldTerminate)
{
	RayState& rayState = rayStates[i];
	if (rayState.terminated == true || rayState.hit == false) { return; }

	Float3 normal = rayState.normal;
	Float3 pos = rayState.pos;
	float rayoffset = rayState.offset;
	Ray ray = rayState.ray;
	Float3 nextRayDir(0, 1, 0);

	Float3& beta = rayState.beta;
	Float3& outColor = rayState.L;

	SurfaceMaterial mat;
	if (rayState.objectIdx == 998) // horizontal plane
	{
		mat.type = PERFECT_REFLECTION;
		mat.albedo = Float3(1, 1, 1);
	}
	else // get materials
	{
		int matIdx = sceneMaterial.materialsIdx[rayState.objectIdx];
		mat = sceneMaterial.materials[matIdx];
	}

	// Is ray into surface? If no, flip normal
	float normalDotRayDir = dot(normal, ray.dir);    // ray dot geometry normal
	bool isRayIntoSurface = normalDotRayDir < 0;     // ray shoot into surface, if dot < 0
	if (isRayIntoSurface == false) { normal = -normal; normalDotRayDir = -normalDotRayDir; } // if ray not shoot into surface, convert the case to "ray shoot into surface" by flip the normal

	if (mat.type == LAMBERTIAN_DIFFUSE)
	{
		LambertianReflection(rd2(&rayState.rdState[0], &rayState.rdState[1]), nextRayDir, normal);
		Float3 surfaceBsdfOverPdf = mat.albedo;
		beta *= surfaceBsdfOverPdf;
	}
	else if (mat.type == PERFECT_REFLECTION)
	{
		nextRayDir = normalize(ray.dir - normal * dot(ray.dir, normal) * 2.0);
	}
	else if (mat.type == MICROFACET_REFLECTION)
	{
		const Float3 F0(0.56, 0.57, 0.58);
		const float alpha = 0.05;
		MacrofacetReflection(rd(&rayState.rdState[0]), rd(&rayState.rdState[1]),
			ray.dir,
			nextRayDir,
			normal,
			beta,
			F0,
			alpha);
	}
	else if (mat.type == PERFECT_FRESNEL_REFLECTION_REFRACTION)
	{
		// eta
		float etaT = 1.33;
		float etaI = 1.0;
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
			nextRayDir = ray.dir - normal * normalDotRayDir * 2.0;
		}
		else
		{
			// Fresnel for dialectric
			float fresnel = FresnelDialetric(etaI, etaT, cosThetaI, cosThetaT);

			// reflection or transmittance
			if (rd(&rayState.rdState[2]) < fresnel)
			{
				nextRayDir = ray.dir - normal * normalDotRayDir * 2.0;
			}
			else
			{
				nextRayDir = eta * ray.dir + (eta * cosThetaI - cosThetaT) * normal;
				rayoffset *= -1.0;
			}
		}

		nextRayDir.normalize();
	}
	else if (mat.type == EMISSIVE)
	{
		outColor += mat.albedo * beta;
		rayState.terminated = true;
	}

	if (rayState.terminated || shouldTerminate)
	{
		rayState.terminated = true;
		Store2D(Float4(outColor, 1.0), colorBuffer, rayState.idx);
	}
	else
	{
		rayState.ray.orig = pos + rayoffset * normal;
		rayState.ray.dir  = nextRayDir;
	}
}

__global__ void RayTraverse2(ConstBuffer cbo, RayState* rayState, SceneGeometry sceneGeometry, uint* uintBufferIn, uint* hitBuffer, uint* missBuffer, uint* hitBufferTop, uint* missBufferTop)
{
	uint i = uintBufferIn[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	RayState& ray = rayState[i];
	if (ray.terminated == true) { return; }
	ray.hit = RaySceneIntersect(ray.ray, sceneGeometry, ray.pos, ray.normal, ray.objectIdx, ray.offset);

	if (ray.hit && ray.terminated == false)
	{
		uint oldVal = atomicInc(hitBufferTop, cbo.bufferSize);
		hitBuffer[oldVal] = i;
	}
	else
	{
		uint oldVal = atomicInc(missBufferTop, cbo.bufferSize);
		missBuffer[oldVal] = i;
	}
}

__global__ void RayTraverse3(ConstBuffer cbo, RayState* rayState, SceneGeometry sceneGeometry, uint* uintBufferIn)
{
	uint i = uintBufferIn[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	RayState& ray = rayState[i];
	if (ray.terminated == true) { return; }
	ray.hit = RaySceneIntersect(ray.ray, sceneGeometry, ray.pos, ray.normal, ray.objectIdx, ray.offset);
}

__global__ void SkyShaderLaunch2(ConstBuffer cbo, RayState* rayStates, SurfObj colorBuffer, uint* uintBuffer)
{
	uint i = uintBuffer[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	Int2 idx(i % cbo.bufferDim.x, i / cbo.bufferDim.x);
	SkyShader(cbo, rayStates, colorBuffer, i, idx);
}

__global__ void SurfaceShaderLaunch2(ConstBuffer cbo, RayState* rayStates, SceneMaterial sceneMaterial, SurfObj colorBuffer, uint* uintBuffer, bool shouldTerminate)
{
	uint i = uintBuffer[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	Int2 idx(i % cbo.bufferDim.x, i / cbo.bufferDim.x);
	SurfaceShader(cbo, rayStates, sceneMaterial, colorBuffer, i, idx, shouldTerminate);
}

__global__ void SkyShaderLaunch(ConstBuffer cbo, RayState* rayStates, SurfObj colorBuffer, uint* bufferTop, uint* idxBuffer) {
	SkyShaderLaunch2<<<dim3(divRoundUp(*bufferTop, 64u), 1, 1), dim3(8, 8, 1), 0>>> (cbo, rayStates, colorBuffer, idxBuffer);
}

__global__ void SurfaceShaderLaunch(ConstBuffer cbo, RayState* rayStates, SceneMaterial sceneMaterial, SurfObj colorBuffer, uint* bufferTop, uint* idxBuffer, bool shouldTerminate = 0) {
	SurfaceShaderLaunch2<<<dim3(divRoundUp(*bufferTop, 64u), 1, 1), dim3(8, 8, 1), 0>>> (cbo, rayStates, sceneMaterial, colorBuffer, idxBuffer, shouldTerminate);
}

__global__ void RayTraverseLaunch(ConstBuffer cbo, RayState* rayStates, SceneGeometry sceneGeometry, uint* bufferTop, uint* idxBuffer, uint* hitBuffer, uint* missBuffer, uint* hitBufferTop, uint* missBufferTop) {
	RayTraverse2<<<dim3(divRoundUp(*bufferTop, 64u), 1, 1), dim3(8, 8, 1), 0>>> (cbo, rayStates, sceneGeometry, idxBuffer, hitBuffer, missBuffer, hitBufferTop, missBufferTop);
}

__global__ void RayTraverseLaunch2(ConstBuffer cbo, RayState* rayStates, SceneGeometry sceneGeometry, uint* bufferTop, uint* idxBuffer) {
	RayTraverse3 <<<dim3(divRoundUp(*bufferTop, 64u), 1, 1), dim3(8, 8, 1), 0 >>> (cbo, rayStates, sceneGeometry, idxBuffer);
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
	d_sceneGeometry.numSphereLights = numSphereLights;
	d_sceneGeometry.sphereLights    = d_sphereLights;

	// camera
	Camera& camera = cbo.camera;
	Float3 cameraLookAtPoint = cameraFocusPos + Float3(0.0f, 0.01f, 0.0f);
	camera.pos               = cameraFocusPos + rotate3f(Float3(0, 1, 0), fmodf(clockTime * TWO_PI / 60, TWO_PI), Float3(0.0f, 0.0f, -0.1f)) + Float3(0, abs(sinf(clockTime * TWO_PI / 60)) * 0.05f, 0);
	camera.dir               = normalize(cameraLookAtPoint - camera.pos.xyz);
	camera.left              = cross(Float3(0, 1, 0), camera.dir.xyz);

	Int2 bufferDim(renderWidth, renderHeight);
	Int2 outputDim(screenWidth, screenHeight);

	// ------------------ Ray Gen -------------------
	indexBuffers.memsetZero(renderBufferSize, streams[0]);
	cudaStreamSynchronize(streams[2]);
	RayGen <<<gridDim, blockDim, 0, streams[1] >>> (cbo, rayState, d_randInitVec, d_sceneGeometry, normalBuffer, positionBuffer, /*out*/indexBuffers);

	cudaStreamSynchronize(streams[0]);
	cudaStreamSynchronize(streams[1]);

	// ------------------ Scene Traverse Loop -------------------
	for (int i = 0; i < 2; ++i)
	{
		//bool terminate = (i == 1);
		bool terminate = false;

		// ----------------- in: A, out: B --------------------
		cudaMemsetAsync(indexBuffers.hitBufferTopB, 0u, sizeof(uint), streams[0]);
		cudaMemsetAsync(indexBuffers.missBufferTopB, 0u, sizeof(uint), streams[0]);

		// sky + surface
		SkyShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[2] >> > (cbo, rayState, colorBufferA,
			/*in*/indexBuffers.missBufferTopA, /*in*/indexBuffers.missBufferA);

		SurfaceShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[3] >> > (cbo, rayState, d_sceneMaterial, colorBufferA,
			/*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA);

		cudaStreamSynchronize(streams[0]);
		cudaStreamSynchronize(streams[2]);
		cudaStreamSynchronize(streams[3]);

		// traverse
		RayTraverseLaunch << < dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[1] >> > (cbo, rayState, d_sceneGeometry,
			/*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA, /*out*/indexBuffers.hitBufferB, /*out*/indexBuffers.missBufferB, /*out*/indexBuffers.hitBufferTopB, /*out*/indexBuffers.missBufferTopB);
		cudaStreamSynchronize(streams[1]);

		// ----------------- in: B, out: A --------------------
		cudaMemsetAsync(indexBuffers.hitBufferTopA, 0u, sizeof(uint), streams[0]);
		cudaMemsetAsync(indexBuffers.missBufferTopA, 0u, sizeof(uint), streams[0]);

		// sky + surface
		SkyShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[2] >> > (cbo, rayState, colorBufferA,
			/*in*/indexBuffers.missBufferTopB, /*in*/indexBuffers.missBufferB);

		SurfaceShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[3] >> > (cbo, rayState, d_sceneMaterial, colorBufferA,
			/*in*/indexBuffers.hitBufferTopB, /*in*/indexBuffers.hitBufferB, /*terminate*/terminate);

		cudaStreamSynchronize(streams[0]);
		cudaStreamSynchronize(streams[2]);
		cudaStreamSynchronize(streams[3]);

		// traverse
		if (terminate == false)
		{
			RayTraverseLaunch << < dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[1] >> > (cbo, rayState, d_sceneGeometry,
				/*in*/indexBuffers.hitBufferTopB, /*in*/indexBuffers.hitBufferB, /*out*/indexBuffers.hitBufferA, /*out*/indexBuffers.missBufferA, /*out*/indexBuffers.hitBufferTopA, /*out*/indexBuffers.missBufferTopA);
			cudaStreamSynchronize(streams[1]);
		}
	}

	for (int i = 0; i < 4; ++i)
	{
		if (i == 0)
		{
			SkyShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[2] >> > (cbo, rayState, colorBufferA, /*in*/indexBuffers.missBufferTopA, /*in*/indexBuffers.missBufferA);
			SurfaceShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[3] >> > (cbo, rayState, d_sceneMaterial, colorBufferA, /*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA);
		}
		else if (i == 3)
		{
			SkyShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[2] >> > (cbo, rayState, colorBufferA, /*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA);
			SurfaceShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[3] >> > (cbo, rayState, d_sceneMaterial, colorBufferA, /*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA, /*terminate*/true);
		}
		else
		{
			SkyShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[2] >> > (cbo, rayState, colorBufferA, /*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA);
			SurfaceShaderLaunch << <dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[3] >> > (cbo, rayState, d_sceneMaterial, colorBufferA, /*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA);
		}
		cudaStreamSynchronize(streams[2]);
		cudaStreamSynchronize(streams[3]);
		if (i != 3)
		{
			RayTraverseLaunch2 << < dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[1] >> > (cbo, rayState, d_sceneGeometry,
				/*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA);
			cudaStreamSynchronize(streams[1]);
		}
	}

	// ---------------- post processing ----------------
	ToneMapping<<<gridDim, blockDim, 0, streams[0]>>>(/*io*/colorBufferA , bufferDim , /*exposure*/1.0);
	Denoise    <<<gridDim, blockDim, 0, streams[0]>>>(/*io*/colorBufferA , /*in*/normalBuffer , /*in*/positionBuffer, bufferDim, cbDenoise);

	if (cbo.frameNum == 1) { BufferCopy<<<gridDim, blockDim, 0, streams[0]>>>(/*out*/colorBufferB , /*in*/colorBufferA , bufferDim); }
	else                   { TAA       <<<gridDim, blockDim, 0, streams[0]>>>(/*io*/colorBufferB  , /*in*/colorBufferA , bufferDim); }

	FilterScale<<<scaleGridDim, scaleBlockDim, 0, streams[0]>>>(/*out*/renderTarget, /*in*/colorBufferB, outputDim, bufferDim);
}