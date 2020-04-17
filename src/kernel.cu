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

__global__ void RayGen(ConstBuffer cbo, RayState* rayState, RandInitVec* randInitVec, SceneGeometry sceneGeometry, SurfObj normalBuffer, SurfObj positionBuffer, ullint* gHitMask, uint randNumUsedPerFrame)
{
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;
	int j = blockDim.x * threadIdx.y + threadIdx.x;
	int k = gridDim.x * blockIdx.y + blockIdx.x;
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

	// init hitmask
	__shared__ ullint hitMask;
	__syncthreads();
	if (j == 0) atomicExch(&hitMask, 0x0);
	__syncthreads();
	// gather hitmask in a block
	ullint bitToSet = (ray.hit ? 1ull : 0ull) << j;
	atomicOr(&hitMask, bitToSet);
	__syncthreads();
	// store shared to global
	if (j == 0) gHitMask[k] = hitMask;
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

__global__ void SkyShaderLaunch(ConstBuffer cbo, RayState* rayStates, SurfObj colorBuffer, uint* uintBuffer)
{
	uint packedOffset = uintBuffer[blockIdx.x];
	Int2 offset((packedOffset & 0x0000ffff) << 3, (packedOffset >> 16) << 3);
	Int2 idx = Int2(threadIdx.x, threadIdx.y) + offset;
	uint i = cbo.gridDim.x * blockDim.x * idx.y + idx.x;
	SkyShader(cbo, rayStates, colorBuffer, i, idx);
}

__global__ void SkyShaderLaunch2(ConstBuffer cbo, RayState* rayStates, SurfObj colorBuffer, uint* uintBuffer)
{
	uint i = uintBuffer[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	Int2 idx(i % cbo.bufferDim.x, i / cbo.bufferDim.x);
	SkyShader(cbo, rayStates, colorBuffer, i, idx);
}

__device__ inline void SurfaceShader(ConstBuffer& cbo, RayState* rayStates, SceneMaterial& sceneMaterial, SurfObj& colorBuffer, uint& i, Int2& idx, bool& shouldTerminate)
{
	RayState& rayState = rayStates[i];
	if (rayState.terminated == true || rayState.hit == false) { return; }

	Float3 normal = rayState.normal;
	Float3 pos = rayState.pos;
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
	if (isRayIntoSurface == false) { normal = -normal; } // if ray not shoot into surface, convert the case to "ray shoot into surface" by flip the normal

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
		rayState.ray.orig = pos + rayState.offset * normal;
		rayState.ray.dir  = nextRayDir;
	}
}

__global__ void SurfaceShaderLaunch(ConstBuffer cbo, RayState* rayStates, SceneMaterial sceneMaterial, SurfObj colorBuffer, uint* uintBuffer, bool shouldTerminate = 0)
{
	uint packedOffset = uintBuffer[blockIdx.x];
	Int2 offset((packedOffset & 0x0000ffff) << 3, (packedOffset >> 16) << 3);
	Int2 idx = Int2(threadIdx.x, threadIdx.y) + offset;
	uint i = cbo.gridDim.x * blockDim.x * idx.y + idx.x;
	SurfaceShader(cbo, rayStates, sceneMaterial, colorBuffer, i, idx, shouldTerminate);
}

__global__ void SurfaceShaderLaunch2(ConstBuffer cbo, RayState* rayStates, SceneMaterial sceneMaterial, SurfObj colorBuffer, uint* uintBuffer, bool shouldTerminate = 0)
{
	uint i = uintBuffer[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	Int2 idx(i % cbo.bufferDim.x, i / cbo.bufferDim.x);
	SurfaceShader(cbo, rayStates, sceneMaterial, colorBuffer, i, idx, shouldTerminate);
}

__global__ void RayTraverse(ConstBuffer cbo, RayState* rayState, SceneGeometry sceneGeometry, uint* uintBufferIn, uint* hitBuffer, uint* missBuffer, uint* hitBufferTop, uint* missBufferTop)
{
	uint packedOffset = uintBufferIn[blockIdx.x];
	Int2 offset((packedOffset & 0x0000ffff) << 3, (packedOffset >> 16) << 3);
	Int2 idx = Int2(threadIdx.x, threadIdx.y) + offset;
	uint i = cbo.gridDim.x * blockDim.x * idx.y + idx.x;

	RayState& ray = rayState[i];
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

__global__ void RayTraverse2(ConstBuffer cbo, RayState* rayState, SceneGeometry sceneGeometry, uint* uintBufferIn, uint* hitBuffer, uint* missBuffer, uint* hitBufferTop, uint* missBufferTop)
{
	uint i = uintBufferIn[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	RayState& ray = rayState[i];
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

__global__ void RayTraverse4(ConstBuffer cbo, RayState* rayState, SceneGeometry sceneGeometry, uint* uintBufferIn)
{
	uint packedOffset = uintBufferIn[blockIdx.x];
	Int2 offset((packedOffset & 0x0000ffff) << 3, (packedOffset >> 16) << 3);
	Int2 idx = Int2(threadIdx.x, threadIdx.y) + offset;
	uint i = cbo.gridDim.x * blockDim.x * idx.y + idx.x;
	RayState& ray = rayState[i];
	if (ray.terminated == true) { return; }
	ray.hit = RaySceneIntersect(ray.ray, sceneGeometry, ray.pos, ray.normal, ray.objectIdx, ray.offset);
}

__global__ void PathTracing(
	ConstBuffer         cbo,
	RayState*           rayStates,
	SurfObj             colorBuffer,
	RandInitVec*        randInitVec,
	SceneGeometry       sceneGeometry,
	SceneMaterial       sceneMaterial,
	ullint*             gHitMask,
	IndexBuffers        indexBuffers)
{
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	uint i = gridDim.x * blockDim.x * idx.y + idx.x;
	if (idx.x >= cbo.gridDim.x || idx.y >= cbo.gridDim.y) return;

	// collect "hit" 8x8 blocks and "miss" 8x8 blocks
	uint packedIdx = (idx.y << 16) | (idx.x & 0x0000ffff);
	if (gHitMask[i] != 0ull) {
		uint oldVal = atomicInc(indexBuffers.hitBufferTopA, cbo.gridSize);
		indexBuffers.hitBufferA[oldVal] = packedIdx;
	}
	if (gHitMask[i] != 0xffffffffffffffffull) {
		uint oldVal = atomicInc(indexBuffers.missBufferTopA, cbo.gridSize);
		indexBuffers.missBufferA[oldVal] = packedIdx;
	}

	if (i == 0) // single thread mission, others are free
	{
		cudaDeviceSynchronize(); // sync 1st

		// launch hit blocks with hitShader, miss blocks with missShader (in block granularity)
		dim3 surfDim(*indexBuffers.hitBufferTopA, 1, 1);
		dim3 skyDim(*indexBuffers.missBufferTopA, 1, 1);
		SkyShaderLaunch<<<skyDim, blockDim>>> (cbo, rayStates, colorBuffer, indexBuffers.missBufferA);
		SurfaceShaderLaunch<<<surfDim, blockDim>>> (cbo, rayStates, sceneMaterial, colorBuffer, indexBuffers.hitBufferA);

		// launch traverse with hit blocks (in block granularity), collect new hit blocks and miss blocks (in thread granularity)
		RayTraverse<<<surfDim, blockDim>>>(cbo, rayStates, sceneGeometry, indexBuffers.hitBufferA, indexBuffers.hitBufferB, indexBuffers.missBufferB, indexBuffers.hitBufferTopB, indexBuffers.missBufferTopB);

		// reset counters
		atomicExch(indexBuffers.hitBufferTopA, 0);
		atomicExch(indexBuffers.missBufferTopA, 0);

		cudaDeviceSynchronize(); // sync 2nd

		// launch hit blocks with hitShader, miss blocks with missShader (in thread granularity)
		surfDim = dim3(divRoundUp(*indexBuffers.hitBufferTopB, 64u), 1, 1);
		skyDim = dim3(divRoundUp(*indexBuffers.missBufferTopB, 64u), 1, 1);

		SkyShaderLaunch2<<<skyDim, blockDim>>> (cbo, rayStates, colorBuffer, indexBuffers.missBufferB);
		SurfaceShaderLaunch2<<<surfDim, blockDim>>> (cbo, rayStates, sceneMaterial, colorBuffer, indexBuffers.hitBufferB);

		// launch traverse with hit blocks (in thread granularity), collect new hit blocks and miss blocks (in thread granularity)
		RayTraverse2<<<surfDim, blockDim>>>(cbo, rayStates, sceneGeometry, indexBuffers.hitBufferB, indexBuffers.hitBufferA, indexBuffers.hitBufferA, indexBuffers.hitBufferTopA, indexBuffers.missBufferTopA);

		cudaDeviceSynchronize(); // sync 3rd

		// launch hit blocks with hitShader, miss blocks with missShader (in thread granularity)
		surfDim = dim3(divRoundUp(*indexBuffers.hitBufferTopA, 64u), 1, 1);
		skyDim = dim3(divRoundUp(*indexBuffers.missBufferTopA, 64u), 1, 1);

		SkyShaderLaunch2<<<skyDim, blockDim>>> (cbo, rayStates, colorBuffer, indexBuffers.hitBufferA);
		SurfaceShaderLaunch2<<<surfDim, blockDim>>> (cbo, rayStates, sceneMaterial, colorBuffer, indexBuffers.hitBufferA);

		// launch with current block setting for multiple times
		for (int bounce = 0; bounce < 4; ++bounce) {
			RayTraverse3<<<surfDim, blockDim>>>(cbo, rayStates, sceneGeometry, indexBuffers.hitBufferA);
			SkyShaderLaunch2<<<surfDim, blockDim>>> (cbo, rayStates, colorBuffer, indexBuffers.hitBufferA);
			SurfaceShaderLaunch2<<<surfDim, blockDim>>> (cbo, rayStates, sceneMaterial, colorBuffer, indexBuffers.hitBufferA);
		}

		// final launch, write to color buffer
		RayTraverse3<<<surfDim, blockDim>>>(cbo, rayStates, sceneGeometry, indexBuffers.hitBufferA);
		SkyShaderLaunch2<<<surfDim, blockDim>>> (cbo, rayStates, colorBuffer, indexBuffers.hitBufferA);
		SurfaceShaderLaunch2<<<surfDim, blockDim>>> (cbo, rayStates, sceneMaterial, colorBuffer, indexBuffers.hitBufferA, true);
	}
}

void RayTracer::draw(SurfObj* renderTarget)
{
	// ---------------- frame constant buffer update ------------------
	timer.update();
	float clockTime   = timer.getTime();

	const Float3 axis = normalize(Float3(0.0, 0.0, 1.0));
	const float angle = fmodf(clockTime * TWO_PI * 0.1f, TWO_PI);
	Float3 sunDir     = rotate3f(axis, angle, Float3(0.0, 1.0, 2.5)).normalized();

	cbo.sunDir        = sunDir;
	cbo.frameNum      = cbo.frameNum + 1;

	indexBuffers.memsetZero(renderBufferSize);

	// ---------------- ray tracing ----------------
	Int2 bufferDim(renderWidth , renderHeight);
	Int2 outputDim(screenWidth , screenHeight);

	RayGen <<<gridDim, blockDim, sizeof(ullint)>>> (cbo, rayState, d_randInitVec, d_sceneGeometry, normalBuffer, positionBuffer, gHitMask, 5);
	PathTracing <<<dim3(divRoundUp(gridDim.x, blockDim.x), divRoundUp(gridDim.y, blockDim.y), 1), blockDim>>> (cbo, rayState, colorBufferA, d_randInitVec, d_sceneGeometry, d_sceneMaterial, gHitMask, indexBuffers);

	//GpuErrorCheck(cudaDeviceSynchronize());
	//GpuErrorCheck(cudaPeekAtLastError());

	// ---------------- post processing ----------------
	ToneMapping<<<gridDim, blockDim>>>(/*io*/colorBufferA , bufferDim , /*exposure*/1.0);
	Denoise    <<<gridDim, blockDim>>>(/*io*/colorBufferA , /*in*/normalBuffer , /*in*/positionBuffer, bufferDim, cbDenoise);

	if (cbo.frameNum == 1) { BufferCopy<<<gridDim, blockDim>>>(/*out*/colorBufferB , /*in*/colorBufferA , bufferDim); }
	else                   { TAA       <<<gridDim, blockDim>>>(/*io*/colorBufferB  , /*in*/colorBufferA , bufferDim); }

	FilterScale<<<scaleGridDim, scaleBlockDim>>>(/*out*/renderTarget, /*in*/colorBufferB, outputDim, bufferDim);
}