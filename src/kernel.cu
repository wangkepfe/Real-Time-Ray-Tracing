#include "kernel.cuh"
#include "debug_util.cuh"
#include "geometry.cuh"
#include "bsdf.cuh"
#include "sampler.cuh"
#include "sky.cuh"
#include "postprocessing.cuh"
#include "raygen.cuh"
#include "traverse.cuh"
#include "light.cuh"

__device__ inline void LightShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, TexObj skyTex)
{
	// check for termination and hit light
	if (rayState.hitLight == false) { return; }

	Float3 beta = rayState.beta;
	Float3 lightDir = rayState.dir;

	// Different light source type
	if (rayState.matType == MAT_SKY)
	{
		// env light
		//Float3 envLightColor = EnvLight(lightDir, cbo.sunDir, cbo.clockTime, rayState.isDiffuseRay);
		//Float3 envLightColor = Float3(0.8f);
		Float3 envLightColor = EnvLight2(lightDir, cbo.clockTime, rayState.isDiffuseRay, skyTex);
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

__device__ inline void GlossyShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, BlueNoiseRandGenerator& randGen, int loopIdx)
{
	// check for termination and hit light
	if (rayState.hitLight == true || rayState.isDiffuse == true) { return; }

	if (loopIdx == 0) { rayState.bounceLimit = 4; }

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
		float surfaceRand = randGen.Rand(4 + loopIdx * 6 + 0);
		PerfectReflectionRefraction(1.0, 1.33, rayState.isRayIntoSurface, rayState.normal, rayState.normalDotRayDir, surfaceRand, rayState.dir, nextRayDir, rayOffset);
		rayState.dir = nextRayDir;
		rayState.orig = rayState.pos + rayOffset * rayState.normal;
	}
}

__device__ inline void DiffuseShader(ConstBuffer& cbo, RayState& rayState, SceneMaterial sceneMaterial, SceneTextures textures, float* skyCdf, BlueNoiseRandGenerator& randGen, int loopIdx)
{
	// check for termination and hit light
	if (rayState.hitLight == true || rayState.isDiffuse == false) { return; }

	rayState.isDiffuseRay = true;

	// get mat
	SurfaceMaterial mat = sceneMaterial.materials[rayState.matId];

	Float3 albedo;
#if 0
	float uvScale = 60.0f;
	if (mat.useTex0)
	{
		float4 texColor = tex2D<float4>(textures.array[mat.texId0], rayState.uv.x * uvScale, rayState.uv.y * uvScale);
		albedo = Float3(texColor.x, texColor.y, texColor.z);
	}
	else
	{
		albedo = mat.albedo;
	}
#else
	albedo = mat.albedo;
#endif

	Float3 normal = rayState.normal;
#if 0
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
#endif

	Float3 rayDir = rayState.dir;

	// light sample
	float isDeltaLight = false;
	Float3 lightSampleDir;
	float lightSamplePdf;
	bool isLightSampled = SampleLight(cbo, rayState, sceneMaterial, lightSampleDir, lightSamplePdf, isDeltaLight, skyCdf, randGen, loopIdx);

	// surface sample
	Float3 surfSampleDir;

	Float3 surfaceBsdfOverPdf;
	Float3 surfaceSampleBsdf;
	float surfaceSamplePdf = 0;

	Float3 lightSampleSurfaceBsdfOverPdf;
	Float3 lightSampleSurfaceBsdf;
	float lightSampleSurfacePdf = 0;

	Float2 surfaceDiffuseRand2 = randGen.Rand2(4 + loopIdx * 6 + 0);
	if (rayState.matType == LAMBERTIAN_DIFFUSE)
	{
		LambertianSample(surfaceDiffuseRand2, surfSampleDir, normal);

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
			MacrofacetReflectionSample(surfaceDiffuseRand2, rayDir, surfSampleDir, normal, surfaceBsdfOverPdf, surfaceSampleBsdf, surfaceSamplePdf, F0, albedo, alpha);

		if (isLightSampled == true)
			MacrofacetReflection(lightSampleSurfaceBsdfOverPdf, lightSampleSurfaceBsdf, lightSampleSurfacePdf, normal, lightSampleDir, rayDir, F0, albedo, alpha);
	}

	// -------------------------------------- MIS balance heuristic ------------------------------------------
	float misRand = randGen.Rand(4 + loopIdx * 6 + 4);
	if (isLightSampled)
	{
		if (isDeltaLight)
		{
			// if a delta light (or say distant/directional light, typically sun light) is sampled,
			// no surface sample is needed since the weight for surface is zero
			rayState.beta *= lightSampleSurfaceBsdf;
			rayState.dir = lightSampleDir;
		}
		else
		{
			// The full equation for MIS is L = sum w_i * f_i / pdf_i
			// which in this case, two samples, one from surface bsdf distribution, one from light distribution
			//
			// L = w_surf * bsdf(dir_surf) / surfaceSamplePdf(dir_surf) + w_light * bsdf(dir_light) / surfaceSamplePdf(dir_light)
			// where w_surf = surfaceSamplePdf(dir_surf) / (surfaceSamplePdf(dir_surf) + lightSamplePdf)
			//       w_light = surfaceSamplePdf(dir_light) / (surfaceSamplePdf(dir_light) + lightSamplePdf)
			//
			// Then it'll become
			// L = bsdf(dir_surf) / (surfaceSamplePdf(dir_surf) + lightSamplePdf) +
			//     bsdf(dir_light) / (surfaceSamplePdf(dir_light) + lightSamplePdf)
			//
			// My algorithm takes bsdf as value and misWeight*pdf as weight,
			// using the weights to choose either sample light or surface.
			// It achieves single ray sample per surface shader with no bias to MIS balance heuristic algorithm
			float lightSampleWeight =  1.0f / (lightSamplePdf + lightSampleSurfacePdf);
			float surfaceSampleWeight = 1.0f / (surfaceSamplePdf + lightSamplePdf);

			float chooseSurfaceFactor = surfaceSampleWeight / (lightSampleWeight + surfaceSampleWeight);

			if (misRand < chooseSurfaceFactor)
			{
				// choose surface scatter sample
				rayState.beta *= min3f(surfaceSampleBsdf * surfaceSampleWeight / chooseSurfaceFactor, Float3(1.0f));
				rayState.dir = surfSampleDir;
			}
			else
			{
				// choose light sample
				rayState.beta *= min3f(lightSampleSurfaceBsdf * lightSampleWeight / (1.0f - chooseSurfaceFactor), Float3(1.0f));
				rayState.dir = lightSampleDir;
			}
		}
	}
	else
	{
		// if no light sample condition is met, sample surface only, which is the vanila case
		rayState.beta *= surfaceBsdfOverPdf;
		rayState.dir = surfSampleDir;
	}

	rayState.orig = rayState.pos + rayState.offset * normal;
}

__global__ void PathTrace(ConstBuffer cbo, SceneGeometry sceneGeometry, SceneMaterial sceneMaterial, BlueNoiseRandGenerator randGen, SurfObj colorBuffer, SceneTextures textures, TexObj skyTex, float* skyCdf)
{
	// index
	Int2 idx(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int i = gridDim.x * blockDim.x * idx.y + idx.x;

	int sampleNum = 1;
	float encodedNormal;
	Float3 outColor = 0;

	#pragma unroll
	for (int sampleIdx = 0; sampleIdx < sampleNum; ++sampleIdx)
	{
		// init ray state
		RayState rayState;
		rayState.i            = i;
		rayState.L            = 0.0;
		rayState.beta         = 1.0;
		rayState.idx          = idx;
		rayState.isDiffuseRay = false;
		rayState.hitLight     = false;
		rayState.bounceLimit  = 2;

		// setup rand gen
		randGen.idx = idx;
		randGen.sampleIdx = cbo.frameNum * sampleNum + sampleIdx;

		// generate ray
		GenerateRay(rayState.orig, rayState.dir, cbo.camera, idx, randGen.Rand2(0), randGen.Rand2(2));
		RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);

		// encode normal
		if (sampleIdx == 0) { encodedNormal = EncodeNormal_R11_G10_B11(rayState.normal); }

		// main loop
		int loopIdx = 0;
		for (; loopIdx < rayState.bounceLimit; ++loopIdx)
		{
			GlossyShader(cbo, rayState, sceneMaterial, randGen, loopIdx);
			DiffuseShader(cbo, rayState, sceneMaterial, textures, skyCdf, randGen, loopIdx);
			RaySceneIntersect(cbo, sceneMaterial, sceneGeometry, rayState);
		}

		GlossyShader(cbo, rayState, sceneMaterial, randGen, loopIdx);
		LightShader(cbo, rayState, sceneMaterial, skyTex);

		outColor += rayState.L;
	}

	// write to buffer
	Store2D(Float4(outColor / (float)sampleNum, encodedNormal), colorBuffer, idx);
}

void RayTracer::UpdateFrame()
{
	// timer
	timer.update();
	deltaTime = timer.getDeltaTime();
	clockTime = timer.getTime();
	cbo.clockTime     = clockTime;

	// sun dir
    const Float3 axis = normalize(Float3(1.0f, 0.0f, -0.4f));
	const float angle = fmodf(clockTime * TWO_PI / 100, TWO_PI);
	sunDir     = rotate3f(axis, angle, Float3(0.0, -1.0, 0.0)).normalized();
	cbo.sunDir        = sunDir;

	// frame number
	static int framen = 1;
	cbo.frameNum      = framen++;

	// prepare for lens flare
	Camera& camera = cbo.camera;
	Mat3 invCamMat(camera.left, camera.up, camera.dir); // build view matrix
	invCamMat.transpose(); // orthogonal matrix, inverse is transpose
	Float3 sunPosViewSpace = sunDir * invCamMat; // transform sun dir to view space
	sunPos = sunPosViewSpace.xy; // get xy
	sunPos /= sunPosViewSpace.z; // get the x and y when z is 1
	sunPos /= camera.tanHalfFov; // [-1, 1]
	sunPos = Float2(0.5) - sunPos * Float2(0.5); // [0, 1]
}

void RayTracer::draw(SurfObj* renderTarget)
{
	// update frame
	UpdateFrame();

	// dimensions
	Int2 bufferDim(renderWidth, renderHeight);
	Int2 outputDim(screenWidth, screenHeight);

	// ---------------- init histogram ----------------
	GpuErrorCheck(cudaMemsetAsync(d_histogram, 0, 64 * sizeof(uint), streams[0]));

	// ---------------- sky ----------------
	Sky<<<dim3(8, 2, 1), dim3(8, 8, 1), 0, streams[0]>>>(skyBuffer, skyCdf, Int2(64, 16), sunDir);
	Scan<<<1, dim3(512, 1, 1), 1024 * sizeof(float), streams[0]>>>(skyCdf, 1024);

	// ---------------- path tracing ----------------------
	PathTrace<<<gridDim, blockDim, 0, streams[0]>>>(cbo, d_sceneGeometry, d_sceneMaterial, d_randGen, colorBufferA, sceneTextures, skyTex, skyCdf);

	DEBUG_CUDA();

	// ---------------- post processing -------------------
	if (cbo.frameNum != 1)
	{
		TemporalFilter <<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA, colorBufferB, bufferDim);
	}

	// Denoise
	AtousFilter<<<dim3(divRoundUp(renderWidth, 28), divRoundUp(renderHeight, 28), 1), dim3(32, 32, 1), 0, streams[0]>>>(colorBufferA, colorBufferB, bufferDim);

	// Histogram
	DownScale4_fp32_fp16<<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA, colorBuffer4, bufferDim);
	DownScale4_fp16_fp16<<<gridDim4, blockDim, 0, streams[0]>>>(colorBuffer4, colorBuffer16, bufferSize4);
	DownScale4_fp16_fp16<<<gridDim16, blockDim, 0, streams[0]>>>(colorBuffer16, colorBuffer64, bufferSize16);

	Histogram2<<<1, dim3(bufferSize64.x, bufferSize64.y, 1), 0, streams[0]>>>(/*out*/d_histogram, /*in*/colorBuffer64 , bufferSize64);

	// Exposure
	AutoExposure<<<1, 1, 0, streams[0]>>>(/*out*/d_exposure, /*in*/d_histogram, (float)(bufferSize64.x * bufferSize64.y), deltaTime);

	// Bloom
	BloomGuassian<<<dim3(divRoundUp(bufferSize4.x, 12), divRoundUp(bufferSize4.y, 12), 1), dim3(16, 16, 1), 0, streams[0]>>>(bloomBuffer4, colorBuffer4, bufferSize4, d_exposure);
	BloomGuassian<<<dim3(divRoundUp(bufferSize16.x, 12), divRoundUp(bufferSize16.y, 12), 1), dim3(16, 16, 1), 0, streams[0]>>>(bloomBuffer16, colorBuffer16, bufferSize16, d_exposure);

	Bloom<<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA, bloomBuffer4, bloomBuffer16, bufferDim, bufferSize4, bufferSize16);

	// Lens flare
	if (sunPos.x > 0 && sunPos.x < 1 && sunPos.y > 0 && sunPos.y < 1 && sunDir.y > -0.0 && dot(sunDir, cbo.camera.dir) > 0)
	{
		sunPos -= Float2(0.5);
		sunPos.x *= (float)renderWidth / (float)renderHeight;
		LensFlare<<<gridDim, blockDim, 0, streams[0]>>>(sunPos, colorBufferA, bufferDim);
	}

	// Tone mapping
	ToneMapping<<<gridDim, blockDim, 0, streams[0]>>>(colorBufferA , bufferDim , d_exposure);

	// Scale to final output
	FilterScale<<<scaleGridDim, scaleBlockDim, 0, streams[0]>>>(/*out*/renderTarget, /*in*/colorBufferA, outputDim, bufferDim);
}