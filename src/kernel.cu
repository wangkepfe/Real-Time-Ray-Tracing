#include "kernel.cuh"
#include "debug_util.cuh"
#include "geometry.cuh"
#include "bsdf.cuh"
#include "sampler.cuh"
#include "sky.cuh"
#include "postprocessing.cuh"
#include "raygen.cuh"
#include "traverse.cuh"
#include "hash.cuh"
#include <thread>

#define TOTAL_LIGHT_MAX_COUNT 3

__device__ inline bool SampleLight(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, Float3& lightSampleDir, float& lightSamplePdf, float& isDeltaLight)
{
	const int numSphereLight = sceneMaterial.numSphereLights;
	Sphere* sphereLights = sceneMaterial.sphereLights;

	float lightChoosePdf;
	int sampledIdx;

	int indexRemap[TOTAL_LIGHT_MAX_COUNT] = {};
	int i = 0;
	int idx = 0;

	const int sunLightIdx = numSphereLight;

	for (; i < numSphereLight; ++i)
	{
		Float3 vec = sphereLights[i].center - rayState.pos;
		if (dot(rayState.normal, vec) > 0 && vec.length2() < 1.0)
			indexRemap[idx++] = i; // sphere light
	}

	if (dot(rayState.normal, cbo.sunDir) > 0.0)
		indexRemap[idx++] = i++; // sun/moon light

	if (idx == 0)
		return false;

	// choose light
	int sampledValue = rd(&rayState.rdState[2]) * idx;
	sampledIdx = indexRemap[sampledValue];
	lightChoosePdf = 1.0 / idx;

	// sample
	if (sampledIdx == sunLightIdx)
	{
		Float3 moonDir = cbo.sunDir;
		moonDir        = -moonDir;
		lightSampleDir = cbo.sunDir.y > -0.05 ? cbo.sunDir : moonDir;
		lightSamplePdf = 1.0f;
		isDeltaLight   = true;
	}
	else
	{
		Sphere sphereLight = sphereLights[sampledIdx];

		Float3 lightDir   = sphereLight.center - rayState.pos;
		float dist2       = lightDir.length2();
		float radius2     = sphereLight.radius * sphereLight.radius;
		float cosThetaMax = sqrtf(max1f(dist2 - radius2, 0)) / sqrtf(dist2);

		Float3 u, v;
		LocalizeSample(lightDir, u, v);
		lightSampleDir = UniformSampleCone(rd2(&rayState.rdState[0], &rayState.rdState[1]), cosThetaMax, u, v, lightDir);
		lightSampleDir = normalize(lightSampleDir);
		lightSamplePdf = UniformConePdf(cosThetaMax) * lightChoosePdf;
	}

	return true;
}

__device__ inline void LightShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial)
{
	// check for termination and hit light
	if ((rayState.terminated == true || rayState.hitLight == false)) { return; }

	Float3 beta = rayState.beta;
	Float3 lightDir = rayState.dir;

	// ray is terminated
	rayState.terminated = true;

	// Different light source type
	if (rayState.matType == MAT_SKY)
	{
		// env light
		Float3 envLightColor = EnvLight(lightDir, cbo.sunDir, cbo.clockTime, rayState.isDiffuseRay);
		rayState.L += envLightColor * beta;
	}
	else if (rayState.matType == EMISSIVE)
	{
		// local light
		SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];
		Float3 L0 = mat.albedo;
		rayState.L += L0 * beta;
	}
}

__device__ inline void GlossyShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial)
{
	// check for termination and hit light
	if (rayState.terminated == true || rayState.hitLight == true || rayState.isDiffuse == true) { return; }

	rayState.hasBsdfRay = true;

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

__device__ inline void DiffuseShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, SceneTextures textures)
{
	// check for termination and hit light
	if (rayState.terminated == true || rayState.hitLight == true || rayState.isDiffuse == false) { return; }

	rayState.hasBsdfRay = true;
	rayState.isDiffuseRay = true;

	// get mat
	SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];

	float uvScale = 60.0f;

	Float3 albedo;
	if (mat.useTex0)
	{
		float4 texColor = tex2D<float4>(textures.array[mat.texId0], rayState.uv.x * uvScale, rayState.uv.y * uvScale);
		albedo = Float3(texColor.x, texColor.y, texColor.z);
	}
	else
	{
		albedo = mat.albedo;
	}

	Float3 normal = rayState.normal;
	if (mat.useTex1)
	{
		float4 texColor = tex2D<float4>(textures.array[mat.texId1], rayState.uv.x * uvScale, rayState.uv.y * uvScale);
		Float3 texNormal = Float3(texColor.x - 0.5, texColor.y - 0.5, texColor.z * 0.5);

		Float3 tangent = Float3(0, 1, 0);

		if (normal.y > 1.0f - 1e-3f)
			tangent = Float3(1, 0, 0);

		Float3 bitangent = cross(normal, tangent);
		tangent = cross(normal, bitangent);

		texNormal = normalize(tangent * texNormal.x + bitangent * texNormal.y + normal * texNormal.z);

		normal = texNormal;
		//normal = mixf(normal, texNormal, 0.0f);
	}

	Float3 rayDir = rayState.dir;

	// light sample
	float isDeltaLight = false;
	Float3 lightSampleDir;
	float lightSamplePdf;
	bool isLightSampled = SampleLight(cbo, rayState, sceneMaterial, lightSampleDir, lightSamplePdf, isDeltaLight);

	// surface sample
	Float3 surfSampleDir;

	Float3 surfaceBsdfOverPdf;
	Float3 surfaceSampleBsdf;
	float surfaceSamplePdf = 0;

	Float3 lightSampleSurfaceBsdfOverPdf;
	Float3 lightSampleSurfaceBsdf;
	float lightSampleSurfacePdf = 0;

	if (rayState.matType == LAMBERTIAN_DIFFUSE)
	{
		LambertianSample(rd2(&rayState.rdState[0], &rayState.rdState[1]), surfSampleDir, normal);

		if (isDeltaLight == false)
		{
			surfaceBsdfOverPdf = LambertianBsdfOverPdf(albedo);
			surfaceSampleBsdf = LambertianBsdf(albedo);
			surfaceSamplePdf = LambertianPdf(surfSampleDir, normal);
		}

		if (isLightSampled == true)
		{
			lightSampleSurfaceBsdfOverPdf = LambertianBsdfOverPdf(albedo);
			lightSampleSurfaceBsdf = LambertianBsdf(albedo);
			lightSampleSurfacePdf = LambertianPdf(lightSampleDir, normal);
		}
	}
	else if (rayState.matType == MICROFACET_REFLECTION)
	{
		Float3 F0 = mat.F0;
		float alpha = mat.alpha;

		if (isDeltaLight == false)
			MacrofacetReflectionSample(rd(&rayState.rdState[0]), rd(&rayState.rdState[1]), rayDir, surfSampleDir, normal, surfaceBsdfOverPdf, surfaceSampleBsdf, surfaceSamplePdf, F0, albedo, alpha);

		if (isLightSampled == true)
			MacrofacetReflection(lightSampleSurfaceBsdfOverPdf, lightSampleSurfaceBsdf, lightSampleSurfacePdf, normal, lightSampleDir, rayDir, F0, albedo, alpha);
	}

	// MIS power/balance heuristic
	Float3 betaWeight;

	if (isLightSampled)
	{
		if (isDeltaLight)
		{
			rayState.beta *= lightSampleSurfaceBsdf;
			rayState.dir = lightSampleDir;
		}
		else
		{
			float lightSampleWeight =  1.0f / (lightSamplePdf + lightSampleSurfacePdf);
			float surfaceSampleWeight = 1.0f / (surfaceSamplePdf + lightSamplePdf);

			bool chooseSurfaceSample = rd(&rayState.rdState[2]) < 0.5;
			/// @todo: surfaceSampleWeight / (lightSampleWeight + surfaceSampleWeight);

			rayState.beta *= chooseSurfaceSample ? (surfaceSampleBsdf * surfaceSampleWeight) : (lightSampleSurfaceBsdf * lightSampleWeight);
			rayState.dir = chooseSurfaceSample ? surfSampleDir : lightSampleDir;
		}
	}
	else
	{
		rayState.beta *= surfaceBsdfOverPdf;
		rayState.dir = surfSampleDir;
	}

	rayState.orig              = rayState.pos + rayState.offset * normal;
}

#if 0
__global__ void PathTrace(ConstBuffer cbo, SceneGeometry sceneGeometry, SceneMaterial sceneMaterial, RandInitVec* randInitVec, SurfObj colorBuffer, SceneTextures textures)
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
	rayState.isDiffuseRay = false;

	// init rand state
	const uint seed = (idx.x << 16) ^ (idx.y);
	for (int k = 0; k < 3; ++k) { curand_init(randInitVec[k], CURAND_2POW32 * 0.5f, WangHash(seed) + cbo.frameNum, &rayState.rdState[k]); }

	// generate ray
	GenerateRay(rayState.orig, rayState.dir, cbo.camera, idx, rd2(&rayState.rdState[0], &rayState.rdState[1]));

	// scene traverse
	RaySceneIntersect(sceneGeometry, rayState);
	UpdateMaterial(cbo, rayState, sceneMaterial);

	rayState.encodedNormal = EncodeNormal_R11_G10_B11(rayState.normal);

	for (int bounce = 0; bounce < 8; ++bounce)
	{
		LightShader(cbo, rayState, sceneMaterial);
		GlossyShader(cbo, rayState, sceneMaterial);
		DiffuseShader(cbo, rayState, sceneMaterial, textures);

		if (bounce != 7)
		{
			if (rayState.terminated == false)
			{
				RaySceneIntersect(sceneGeometry, rayState);
				UpdateMaterial(cbo, rayState, sceneMaterial);
			}
		}
	}

	// write to buffer
	Store2D(Float4(rayState.L, rayState.encodedNormal), colorBuffer, idx);
}

__global__ void SurfaceShaderTraverse(RayState* rayStates, ConstBuffer cbo, SceneGeometry sceneGeometry, SceneMaterial sceneMaterial, SurfObj colorBuffer, SceneTextures textures)
{
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;
	RayState& rayState = rayStates[i];

	for (int bounce = 0; bounce < 8; ++bounce)
	{
		LightShader(cbo, rayState, sceneMaterial);
		GlossyShader(cbo, rayState, sceneMaterial);
		DiffuseShader(cbo, rayState, sceneMaterial, textures);

		if (bounce != 7)
		{
			if (rayState.terminated == false)
			{
				RaySceneIntersect(sceneGeometry, rayState);
				UpdateMaterial(cbo, rayState, sceneMaterial);
			}
		}
	}

	Store2D(Float4(rayState.L, rayState.encodedNormal), colorBuffer, idx);
}

__global__ void RayTraverse(RayState* rayStates, ConstBuffer cbo, SceneGeometry sceneGeometry, SceneMaterial sceneMaterial)
{
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;
	RayState& rayState = rayStates[i];

	if (rayState.terminated == false)
	{
		RaySceneIntersect(sceneGeometry, rayState);
		UpdateMaterial(cbo, rayState, sceneMaterial);
	}
}

__global__ void SurfaceShader(RayState* rayStates, ConstBuffer cbo, SceneMaterial sceneMaterial, SurfObj colorBuffer, SceneTextures textures, bool storeColor = false)
{
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;
	RayState& rayState = rayStates[i];

	GlossyShader(cbo, rayState, sceneMaterial);
	DiffuseShader(cbo, rayState, sceneMaterial, textures);

	if (storeColor)
		Store2D(Float4(rayState.L, rayState.encodedNormal), colorBuffer, idx);
}

__global__ void LightShader1(RayState* rayStates, ConstBuffer cbo, SceneMaterial sceneMaterial, SurfObj colorBuffer, bool storeColor = false)
{
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;
	RayState& rayState = rayStates[i];

	LightShader(cbo, rayState, sceneMaterial);

	if (storeColor)
		Store2D(Float4(rayState.L, rayState.encodedNormal), colorBuffer, idx);
}
#endif

__global__ void RayGenTraverse(RayState* rayStates, ConstBuffer cbo, SceneGeometry sceneGeometry, SceneMaterial sceneMaterial, RandInitVec* randInitVec, IndexBuffers indexBuffers)
{
	// index
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;

	// init ray state
	RayState& rayState    = rayStates[i];
	rayState.i            = i;
	rayState.L            = 0.0;
	rayState.beta         = 1.0;
	rayState.idx          = idx;
	rayState.bounce       = 0;
	rayState.terminated   = false;
	rayState.isDiffuseRay = false;

	// init rand state
	const uint seed = (idx.x << 16) ^ (idx.y);
	for (int k = 0; k < 3; ++k) { curand_init(randInitVec[k], CURAND_2POW32 * 0.5f, WangHash(seed) + cbo.frameNum, &rayState.rdState[k]); }

	// generate ray
	GenerateRay(rayState.orig, rayState.dir, cbo.camera, idx, rd2(&rayState.rdState[0], &rayState.rdState[1]));

	// scene traverse
	RaySceneIntersect(sceneGeometry, rayState);
	UpdateMaterial(cbo, rayState, sceneMaterial);

	rayState.encodedNormal = EncodeNormal_R11_G10_B11(rayState.normal);

	// regroup
	uint* hitBuffer = indexBuffers.hitBufferA;
	uint* missBuffer = indexBuffers.missBufferA;
	uint* hitBufferTop = indexBuffers.hitBufferTopA;
	uint* missBufferTop = indexBuffers.missBufferTopA;
	uint oldVal;

	if (rayState.hitLight == false)
	{
		oldVal = atomicInc(hitBufferTop, cbo.bufferSize);
		hitBuffer[oldVal] = i;
	}
	else
	{
		oldVal = atomicInc(missBufferTop, cbo.bufferSize);
		missBuffer[oldVal] = i;
	}

	//printf("i = %d, hitLight = %d, oldVal = %d\n", i, rayState.hitLight, oldVal);
}


__global__ void RayTraverse2(uint* idxBuffer, RayState* rayStates, ConstBuffer cbo, SceneGeometry sceneGeometry, SceneMaterial sceneMaterial, uint* hitBuffer, uint* missBuffer, uint* hitBufferTop, uint* missBufferTop)
{
	uint i = idxBuffer[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	RayState& rayState = rayStates[i];

	if (rayState.terminated == false)
	{
		RaySceneIntersect(sceneGeometry, rayState);
		UpdateMaterial(cbo, rayState, sceneMaterial);
	}

	if ((rayState.hitLight == false) && (rayState.terminated == false))
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

__global__ void SurfaceShader2(uint* idxBuffer, RayState* rayStates, ConstBuffer cbo, SceneMaterial sceneMaterial, SceneTextures textures)
{
	uint i = idxBuffer[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	RayState& rayState = rayStates[i];
	if (rayState.terminated == true) return;

	GlossyShader(cbo, rayState, sceneMaterial);
	DiffuseShader(cbo, rayState, sceneMaterial, textures);
}

__global__ void LightShader2(uint* idxBuffer, RayState* rayStates, ConstBuffer cbo, SceneMaterial sceneMaterial, SurfObj colorBuffer)
{
	uint i = idxBuffer[blockIdx.x * 64 + threadIdx.y * 8 + threadIdx.x];
	RayState& rayState = rayStates[i];
	if (rayState.terminated == true) return;

	LightShader(cbo, rayState, sceneMaterial);

	Int2 idx(i % cbo.bufferDim.x, i / cbo.bufferDim.x);
	Store2D(Float4(rayState.L, rayState.encodedNormal), colorBuffer, idx);
}

__global__ void LightShaderLaunch(
	uint*         bufferTop,
	uint*         idxBuffer,

	RayState*     rayStates,
	ConstBuffer   cbo,
	SceneMaterial sceneMaterial,
	SurfObj       colorBuffer)
{
	LightShader2<<<dim3(divRoundUp(*bufferTop, 64u), 1, 1), dim3(8, 8, 1), 0>>> (
		idxBuffer,
		rayStates, cbo, sceneMaterial, colorBuffer);
}

__global__ void SurfaceShaderLaunch(
	uint*         bufferTop,
	uint*         idxBuffer,

	RayState*     rayStates,
	ConstBuffer   cbo,
	SceneMaterial sceneMaterial,
	SceneTextures textures)
{
	SurfaceShader2<<<dim3(divRoundUp(*bufferTop, 64u), 1, 1), dim3(8, 8, 1), 0>>> (
		idxBuffer,
		rayStates, cbo, sceneMaterial, textures);
}

__global__ void RayTraverseLaunch(
	uint*         bufferTop,
	uint*         idxBuffer,

	RayState*     rayStates,
	ConstBuffer   cbo,
	SceneGeometry sceneGeometry,
	SceneMaterial sceneMaterial,

	uint*         hitBuffer,
	uint*         missBuffer,
	uint*         hitBufferTop,
	uint*         missBufferTop)
{
	RayTraverse2<<<dim3(divRoundUp(*bufferTop, 64u), 1, 1), dim3(8, 8, 1), 0>>> (
		idxBuffer,
		rayStates, cbo, sceneGeometry, sceneMaterial,
		hitBuffer, missBuffer, hitBufferTop, missBufferTop);
}

void RayTracer::draw(SurfObj* renderTarget)
{
	// ---------------- frame update ------------------
	timer.update();
	float deltaTime = timer.getDeltaTime();
	float clockTime = timer.getTime();

	cbo.clockTime     = clockTime;

    const Float3 axis = normalize(Float3(1.0f, 0.0f, -0.4f));
	const float angle = fmodf(clockTime * TWO_PI / 100, TWO_PI);
	Float3 sunDir     = rotate3f(axis, angle, Float3(0.0, 1.0, 0.0)).normalized();

	cbo.sunDir        = sunDir;
	cbo.frameNum      = cbo.frameNum + 1;

	// camera
	Camera& camera = cbo.camera;

	Float3 camUp = normalize(cross(camera.dir.xyz, camera.left.xyz)); // up
	Mat3 invCamMat(camera.left.xyz, camUp, camera.dir.xyz); // build view matrix
	invCamMat.transpose(); // orthogonal matrix, inverse is transpose
	Float3 sunPosViewSpace = sunDir * invCamMat; // transform sun dir to view space
	Float2 sunPos = sunPosViewSpace.xy; // get xy
	sunPos /= sunPosViewSpace.z; // get the x and y when z is 1
	sunPos /= camera.fov.zw; // [-1, 1]
	sunPos = Float2(0.5) - sunPos * Float2(0.5); // [0, 1]

	Int2 bufferDim(renderWidth, renderHeight);
	Int2 outputDim(screenWidth, screenHeight);

	GpuErrorCheck(cudaMemsetAsync(d_histogram, 0, 64 * sizeof(uint), streams[0]));

	// ------------------ Ray Gen -------------------
#if 0 // 41 FPS
	PathTrace<<<gridDim, blockDim, 0, streams[0]>>>(cbo, d_sceneGeometry, d_sceneMaterial, d_randInitVec, colorBufferA, sceneTextures);
#elif 0 // 21 FPS
	RayGenTraverse<<<gridDim, blockDim, 0, streams[0]>>>(rayStates, cbo, d_sceneGeometry, d_sceneMaterial, d_randInitVec);
	SurfaceShaderTraverse<<<gridDim, blockDim, 0, streams[0]>>>(rayStates, cbo, d_sceneGeometry, d_sceneMaterial, colorBufferA, sceneTextures);
#elif 0 // 19 FPS
	RayGenTraverse<<<gridDim, blockDim, 0, streams[0]>>>(rayStates, cbo, d_sceneGeometry, d_sceneMaterial, d_randInitVec);
	for (int bounce = 0; bounce < 8; ++bounce)
	{
		if (bounce != 7)
		{
			SurfaceShader<<<gridDim, blockDim, 0, streams[0]>>>(rayStates, cbo, d_sceneGeometry, d_sceneMaterial, colorBufferA, sceneTextures);
			RayTraverse<<<gridDim, blockDim, 0, streams[0]>>>(rayStates, cbo, d_sceneGeometry, d_sceneMaterial);
		}
		else
		{
			SurfaceShader<<<gridDim, blockDim, 0, streams[0]>>>(rayStates, cbo, d_sceneGeometry, d_sceneMaterial, colorBufferA, sceneTextures, true);
		}
	}
#elif 1
	indexBuffers.memsetZero(renderBufferSize, streams[0]);

	RayGenTraverse<<<gridDim, blockDim, 0, streams[0]>>>(rayStates, cbo, d_sceneGeometry, d_sceneMaterial, d_randInitVec, indexBuffers);

	cudaStreamSynchronize(streams[0]);

	SurfaceShaderLaunch<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[1]>>>(
	/*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA,
	rayStates, cbo, d_sceneMaterial, sceneTextures);

	LightShaderLaunch<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[0]>>>(
	/*in*/indexBuffers.missBufferTopA, /*in*/indexBuffers.missBufferA,
	rayStates, cbo, d_sceneMaterial, colorBufferA);

	cudaStreamSynchronize(streams[0]);
	cudaStreamSynchronize(streams[1]);

	RayTraverseLaunch<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[0]>>>(
		/*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA,
		rayStates, cbo, d_sceneGeometry, d_sceneMaterial,
		/*out*/indexBuffers.hitBufferB, /*out*/indexBuffers.missBufferB,
		/*out*/indexBuffers.hitBufferTopB, /*out*/indexBuffers.missBufferTopB);

	LightShaderLaunch<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, streams[0]>>>(
		/*in*/indexBuffers.missBufferTopB, /*in*/indexBuffers.missBufferB,
		rayStates, cbo, d_sceneMaterial, colorBufferA);

	//SurfaceShaderLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//		/*in*/indexBuffers.hitBufferTopB, /*in*/indexBuffers.hitBufferB,
	//		rayStates, cbo, d_sceneMaterial, colorBufferA, sceneTextures, true);

	//GpuErrorCheck(cudaDeviceSynchronize());
	//GpuErrorCheck(cudaPeekAtLastError());

	//for (int bounce = 0; bounce < 2; ++bounce)
	//{
	//	if (bounce != 1)
	//	{
	//		if (bounce % 2 == 0)
	//		{
	//			cudaMemsetAsync(indexBuffers.hitBufferTopB, 0u, sizeof(uint), streams[0]);
	//			cudaMemsetAsync(indexBuffers.missBufferTopB, 0u, sizeof(uint), streams[0]);

	//			LightShaderLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.missBufferTopA, /*in*/indexBuffers.missBufferA,
	//				rayStates, cbo, d_sceneMaterial, colorBufferA);

	//			GpuErrorCheck(cudaDeviceSynchronize());
	//			GpuErrorCheck(cudaPeekAtLastError());

	//			SurfaceShaderLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA,
	//				rayStates, cbo, d_sceneMaterial, colorBufferA, sceneTextures);

	//			GpuErrorCheck(cudaDeviceSynchronize());
	//			GpuErrorCheck(cudaPeekAtLastError());

	//			RayTraverseLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.hitBufferTopA, /*in*/indexBuffers.hitBufferA,
	//				rayStates, cbo, d_sceneGeometry, d_sceneMaterial,
	//				/*out*/indexBuffers.hitBufferB, /*out*/indexBuffers.missBufferB,
	//				/*out*/indexBuffers.hitBufferTopB, /*out*/indexBuffers.missBufferTopB);

	//			GpuErrorCheck(cudaDeviceSynchronize());
	//			GpuErrorCheck(cudaPeekAtLastError());
	//		}
	//		else
	//		{
	//			cudaMemsetAsync(indexBuffers.hitBufferTopA, 0u, sizeof(uint), streams[0]);
	//			cudaMemsetAsync(indexBuffers.missBufferTopA, 0u, sizeof(uint), streams[0]);

	//			LightShaderLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.missBufferTopB, /*in*/indexBuffers.missBufferB,
	//				rayStates, cbo, d_sceneMaterial, colorBufferA);

	//			GpuErrorCheck(cudaDeviceSynchronize());
	//			GpuErrorCheck(cudaPeekAtLastError());

	//			SurfaceShaderLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.hitBufferTopB, /*in*/indexBuffers.hitBufferB,
	//				rayStates, cbo, d_sceneMaterial, colorBufferA, sceneTextures);

	//			GpuErrorCheck(cudaDeviceSynchronize());
	//			GpuErrorCheck(cudaPeekAtLastError());

	//			RayTraverseLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.hitBufferTopB, /*in*/indexBuffers.hitBufferB,
	//				rayStates, cbo, d_sceneGeometry, d_sceneMaterial,
	//				/*out*/indexBuffers.hitBufferA, /*out*/indexBuffers.missBufferA,
	//				/*out*/indexBuffers.hitBufferTopA, /*out*/indexBuffers.missBufferTopA);

	//			GpuErrorCheck(cudaDeviceSynchronize());
	//			GpuErrorCheck(cudaPeekAtLastError());
	//		}
	//	}
	//	else
	//	{
	//		LightShaderLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.missBufferTopB, /*in*/indexBuffers.missBufferB,
	//				rayStates, cbo, d_sceneMaterial, colorBufferA, true);

	//		GpuErrorCheck(cudaDeviceSynchronize());
	//		GpuErrorCheck(cudaPeekAtLastError());

	//		SurfaceShaderLaunch<<<gridDim, blockDim, 0, streams[0]>>>(
	//				/*in*/indexBuffers.hitBufferTopB, /*in*/indexBuffers.hitBufferB,
	//				rayStates, cbo, d_sceneMaterial, colorBufferA, sceneTextures, true);

	//		GpuErrorCheck(cudaDeviceSynchronize());
	//		GpuErrorCheck(cudaPeekAtLastError());
	//	}
	//}
#endif

	// ---------------- post processing ----------------

	//if (cbo.frameNum == 1) { BufferCopy<<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA, colorBufferB, bufferDim); }
	//else                   { TAA       <<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA, colorBufferB, bufferDim); }

	// DenoiseKernel<<<dim3(divRoundUp(renderWidth, 28), divRoundUp(renderHeight, 28), 1), dim3(32, 32, 1), 0, streams[0]>>>(colorBufferA, bufferDim);

	// DownScale4<<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA, colorBuffer4, bufferDim);
	// DownScale4<<<gridDim4, blockDim, 0, streams[0]>>>(colorBuffer4, colorBuffer16, bufferSize4);
	// DownScale4<<<gridDim16, blockDim, 0, streams[0]>>>(colorBuffer16, colorBuffer64, bufferSize16);

	// Histogram2<<<1, dim3(bufferSize64.x, bufferSize64.y, 1), 0, streams[0]>>>(/*out*/d_histogram, /*in*/colorBuffer64 , bufferSize64);

	// AutoExposure<<<1, 1, 0, streams[0]>>>(/*out*/d_exposure, /*in*/d_histogram, (float)(bufferSize64.x * bufferSize64.y), deltaTime);

	// BloomGuassian<<<dim3(divRoundUp(bufferSize4.x, 12), divRoundUp(bufferSize4.y, 12), 1), dim3(16, 16, 1), 0, streams[0]>>>(bloomBuffer4, colorBuffer4, bufferSize4, d_exposure);
	// BloomGuassian<<<dim3(divRoundUp(bufferSize16.x, 12), divRoundUp(bufferSize16.y, 12), 1), dim3(16, 16, 1), 0, streams[0]>>>(bloomBuffer16, colorBuffer16, bufferSize16, d_exposure);

	// Bloom<<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA, bloomBuffer4, bloomBuffer16, bufferDim, bufferSize4, bufferSize16);

	// if (sunPos.x > 0 && sunPos.x < 1 && sunPos.y > 0 && sunPos.y < 1 && sunDir.y > -0.02 && dot(sunDir, camera.dir.xyz) > 0)
	// {
	// 	sunPos -= Float2(0.5);
	// 	sunPos.x *= (float)renderWidth / (float)renderHeight;
	// 	LensFlare<<<gridDim, blockDim, 0, streams[0]>>>(sunPos, colorBufferA, bufferDim);
	// }

	ToneMapping<<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA , bufferDim , d_exposure);

	FilterScale<<<scaleGridDim, scaleBlockDim, 0, streams[0]>>>(/*out*/renderTarget, /*in*/colorBufferA, outputDim, bufferDim);

	//FilterScale<<<scaleGridDim, scaleBlockDim, 0, streams[0]>>>(/*out*/renderTarget, /*in*/bloomBuffer4, outputDim, bufferSize4);
}