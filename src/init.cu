
#include "kernel.cuh"
#include "fileUtils.cuh"
#include "blueNoiseRandGenData.h"
#include "cuda_fp16.h"

template<typename T>
__global__ void InitBuffer(T val, SurfObj buffer, Int2 bufferSize)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= bufferSize.x || idx.y >= bufferSize.y) return;

	surf2Dwrite(val, buffer, idx.x, idx.y, cudaBoundaryModeClamp);
}

void RayTracer::init(cudaStream_t* cudaStreams)
{
	maxRenderWidth = 3840;
	maxRenderHeight = 2160;

	if (UseDynamicResolution)
	{
		renderWidth = maxRenderWidth;
		renderHeight = maxRenderHeight;
	}
	else
	{
		renderWidth = screenWidth;
		renderHeight = screenHeight;
	}

	uint i;

	// set streams
	streams = cudaStreams;

	// init cuda
	gpuDeviceInit(0);

	// scope for shorter cpu buffer lifetime
	{
		// load triangles
		std::vector<Triangle> h_triangles;
		//const chat* filename = "resources/models/testCube.obj";
		const char* filename = "resources/models/monkey.obj";
		LoadScene(filename, h_triangles);
		triCount = static_cast<uint>(h_triangles.size());

		GpuErrorCheck(cudaMalloc((void**)& constTriangles, triCount * sizeof(Triangle)));
		GpuErrorCheck(cudaMemcpy(constTriangles, h_triangles.data(), triCount * sizeof(Triangle), cudaMemcpyHostToDevice));
	}

	// bvh
	GpuErrorCheck(cudaMalloc((void**)& triangles, triCount * sizeof(Triangle)));
	GpuErrorCheck(cudaMalloc((void**)& aabbs, triCount * sizeof(AABB)));

	GpuErrorCheck(cudaMalloc((void**) &sceneBoundingBox, sizeof(AABB)));

	GpuErrorCheck(cudaMalloc((void**)& morton, BVHcapacity * sizeof(uint)));
	GpuErrorCheck(cudaMemset(morton, UINT_MAX, BVHcapacity * sizeof(uint)));

	GpuErrorCheck(cudaMalloc((void**)& reorderIdx, BVHcapacity * sizeof(uint)));

	GpuErrorCheck(cudaMalloc((void**)& bvhNodes, (triCount - 1) * sizeof(BVHNode)));
	GpuErrorCheck(cudaMalloc((void**)& isAabbDone, (triCount - 1) * sizeof(uint)));
	GpuErrorCheck(cudaMemset(isAabbDone, 0, (triCount - 1) * sizeof(uint)));

	// AABB
	int numAabbs = 2;
	sceneAabbs = new AABB[numAabbs];
	i = 0;
	sceneAabbs[i++] = AABB({0.0f, 0.0f, 0.0f}, 0.01f);
	sceneAabbs[i++] = AABB({0.0f, 0.0f, 0.0f}, 0.01f);

	// sphere
	numSpheres = 2;
	spheres    = new Sphere[numSpheres];
	i = 0;
	spheres[i++] = Sphere({0.0f, 1.0f, 4.0f}, 1.0f);
	spheres[i++] = Sphere({0.0f, 1.0f, -4.0f}, 1.0f);

	// surface materials
	const int numMaterials     = 10;
	SurfaceMaterial* materials = new SurfaceMaterial[numMaterials];
	i = 0;
	materials[i].type          = EMISSIVE;
	materials[i].albedo        = Float3(0.1f, 0.2f, 0.9f);
	++i;
	materials[i].type          = PERFECT_FRESNEL_REFLECTION_REFRACTION;
	++i;
	materials[i].type          = EMISSIVE;
	materials[i].albedo        = Float3(0.9f, 0.2f, 0.1f);
	++i;
	materials[i].type          = LAMBERTIAN_DIFFUSE;
	materials[i].albedo        = Float3(0.9f);
	++i;
	materials[i].type          = MICROFACET_REFLECTION;
	materials[i].albedo        = Float3(0.9f);
	materials[i].F0            = Float3(0.56f, 0.57f, 0.58f);
	materials[i].alpha         = 0.05f;
	++i;
	materials[i].type          = MICROFACET_REFLECTION;
	materials[i].albedo        = Float3(0.9f);
	materials[i].F0            = Float3(0.56f, 0.57f, 0.58f);
	materials[i].alpha         = 0.01f;
	++i;
	materials[i].type          = LAMBERTIAN_DIFFUSE;
	materials[i].useTex0       = true;
	materials[i].texId0        = 0;
	++i;
	materials[i].type          = LAMBERTIAN_DIFFUSE;
	materials[i].albedo        = Float3(0.9f, 0.2f, 0.1f);
	++i;
	materials[i].type          = LAMBERTIAN_DIFFUSE;
	materials[i].albedo        = Float3(0.2f, 0.9f, 0.1f);
	++i;
	materials[i].type          = LAMBERTIAN_DIFFUSE;
	materials[i].albedo        = Float3(0.1f, 0.2f, 0.9f);


	// number of objects
	const int numObjects = triCount + numSpheres;

	// material index
	int* materialsIdx = new int[numObjects];
	for (i = 0; i < triCount; ++i)
	{
		materialsIdx[i] = 1;
	}
	materialsIdx[i++] = 0;
	materialsIdx[i++] = 2;

	// light source
	numSphereLights = 2;
	sphereLights = new Sphere[numSphereLights];
	for (i = 0; i < numSphereLights; ++i)
	{
		sphereLights[i] = spheres[i];
	}

	// constant buffer
	cbo.frameNum = 0;
	cbo.bvhDebugLevel = -1;

	// launch param
	blockDim = dim3(8, 8, 1);
	gridDim = dim3(divRoundUp(renderWidth, blockDim.x), divRoundUp(renderHeight, blockDim.y), 1);

	scaleBlockDim = dim3(8, 8, 1);
	scaleGridDim = dim3(divRoundUp(screenWidth, scaleBlockDim.x), divRoundUp(screenHeight, scaleBlockDim.y), 1);

	// ------------------------ surface/texture object ---------------------------
	cudaChannelFormatDesc format_color_RGB16_mask_A16 = cudaCreateChannelDescHalf4();
	cudaChannelFormatDesc format_normal_R11_G10_B11_depth_R32 = cudaCreateChannelDesc<float2>();
	cudaChannelFormatDesc format_motionVector_UV16 = cudaCreateChannelDescHalf2();
	cudaChannelFormatDesc format_sampleCount_R8 = cudaCreateChannelDesc<uchar1>();
	cudaChannelFormatDesc format_noiseLevel_R16 = cudaCreateChannelDescHalf1();

	// resource desription
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;

	// texture description
	cudaTextureDesc texDesc  = {};
	texDesc.addressMode[0]   = cudaAddressModeClamp;
	texDesc.addressMode[1]   = cudaAddressModeClamp;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// array A: main render buffer
	GpuErrorCheck(cudaMallocArray(&colorBufferArrayA, &format_color_RGB16_mask_A16, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArrayA;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferA, &resDesc));

	// array B: TAA buffer
	GpuErrorCheck(cudaMallocArray(&colorBufferArrayB, &format_color_RGB16_mask_A16, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArrayB;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferB, &resDesc));

	// normal depth
	GpuErrorCheck(cudaMallocArray(&normalDepthBufferArrayA, &format_normal_R11_G10_B11_depth_R32, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = normalDepthBufferArrayA;
	GpuErrorCheck(cudaCreateSurfaceObject(&normalDepthBufferA, &resDesc));

	GpuErrorCheck(cudaMallocArray(&normalDepthBufferArrayB, &format_normal_R11_G10_B11_depth_R32, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = normalDepthBufferArrayB;
	GpuErrorCheck(cudaCreateSurfaceObject(&normalDepthBufferB, &resDesc));

	// motion vector buffer
	GpuErrorCheck(cudaMallocArray(&motionVectorBufferArray, &format_motionVector_UV16, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = motionVectorBufferArray;
	GpuErrorCheck(cudaCreateSurfaceObject(&motionVectorBuffer, &resDesc));

	// sample count buffer
	GpuErrorCheck(cudaMallocArray(&sampleCountBufferArray, &format_sampleCount_R8, gridDim.x, gridDim.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = sampleCountBufferArray;
	GpuErrorCheck(cudaCreateSurfaceObject(&sampleCountBuffer, &resDesc));

	InitBuffer<<<dim3(divRoundUp(gridDim.x, 8), divRoundUp(gridDim.y, 8), 1), dim3(8, 8, 1)>>> (make_uchar1(1), sampleCountBuffer, Int2(gridDim.x, gridDim.y));

	// noise level uffer
	GpuErrorCheck(cudaMallocArray(&noiseLevelBufferArray, &format_noiseLevel_R16, gridDim.x, gridDim.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = noiseLevelBufferArray;
	GpuErrorCheck(cudaCreateSurfaceObject(&noiseLevelBuffer, &resDesc));

	InitBuffer<<<dim3(divRoundUp(gridDim.x, 8), divRoundUp(gridDim.y, 8), 1), dim3(8, 8, 1)>>> (make_ushort1(0), noiseLevelBuffer, Int2(gridDim.x, gridDim.y));

	// color buffer 1/4 size
	bufferSize4 = UInt2(divRoundUp(renderWidth, 4u), divRoundUp(renderHeight, 4u));
	gridDim4 = dim3(divRoundUp(bufferSize4.x, blockDim.x), divRoundUp(bufferSize4.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray4, &format_color_RGB16_mask_A16, bufferSize4.x, bufferSize4.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArray4;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer4, &resDesc));

	// bloom buffer 1/4 size
	GpuErrorCheck(cudaMallocArray(&bloomBufferArray4, &format_color_RGB16_mask_A16, bufferSize4.x, bufferSize4.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = bloomBufferArray4;
	GpuErrorCheck(cudaCreateSurfaceObject(&bloomBuffer4, &resDesc));

	// color buffer 1/16 size
	bufferSize16 = UInt2(divRoundUp(bufferSize4.x, 4u), divRoundUp(bufferSize4.y, 4u));
	gridDim16 = dim3(divRoundUp(bufferSize16.x, blockDim.x), divRoundUp(bufferSize16.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray16, &format_color_RGB16_mask_A16, bufferSize16.x, bufferSize16.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArray16;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer16, &resDesc));

	// bloom buffer 1/16 size
	GpuErrorCheck(cudaMallocArray(&bloomBufferArray16, &format_color_RGB16_mask_A16, bufferSize16.x, bufferSize16.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = bloomBufferArray16;
	GpuErrorCheck(cudaCreateSurfaceObject(&bloomBuffer16, &resDesc));

	// color buffer 1/64 size
	bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u), divRoundUp(bufferSize16.y, 4u));
	gridDim64 = dim3(divRoundUp(bufferSize64.x, blockDim.x), divRoundUp(bufferSize64.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray64, &format_color_RGB16_mask_A16, bufferSize64.x, bufferSize64.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArray64;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer64, &resDesc));

	// ----------------------- sky buffer ------------------------
	GpuErrorCheck(cudaMallocArray(&skyArray, &format_color_RGB16_mask_A16, skyWidth, skyHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = skyArray;
	GpuErrorCheck(cudaCreateSurfaceObject(&skyBuffer, &resDesc));
	GpuErrorCheck(cudaCreateTextureObject(&skyTex, &resDesc, &texDesc, NULL));

	GpuErrorCheck(cudaMalloc((void**)&skyCdf, skySize * sizeof(float)));
	GpuErrorCheck(cudaMemset(skyCdf, 0, skySize * sizeof(float)));

	// ----------------------- GPU buffers -----------------------
	// exposure
	GpuErrorCheck(cudaMalloc((void**)& d_exposure, 4 * sizeof(float)));
	float initExposureLum[4] = { 1.0f, 1.0f, 1.0f, 1.0f }; // (exposureValue, historyAverageLuminance, historyBrightThresholdLuminance, unused)
	GpuErrorCheck(cudaMemcpy(d_exposure, initExposureLum, 4 * sizeof(float), cudaMemcpyHostToDevice));

	// histogram
	GpuErrorCheck(cudaMalloc((void**)& d_histogram, 64 * sizeof(uint)));

	// scene
	GpuErrorCheck(cudaMalloc((void**)& d_spheres          , numSpheres *       sizeof(Sphere)));
	GpuErrorCheck(cudaMalloc((void**)& d_sceneAabbs       , numAabbs *         sizeof(AABB)));
	GpuErrorCheck(cudaMalloc((void**)& d_materialsIdx     , numObjects *       sizeof(int)));
	GpuErrorCheck(cudaMalloc((void**)& d_surfaceMaterials , numMaterials *     sizeof(SurfaceMaterial)));
	GpuErrorCheck(cudaMalloc((void**)& d_sphereLights     , numSphereLights *  sizeof(Float4)));

	GpuErrorCheck(cudaMemcpy(d_spheres          , spheres      , numSpheres *      sizeof(Float4)         , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_surfaceMaterials , materials    , numMaterials *    sizeof(SurfaceMaterial), cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_sceneAabbs       , sceneAabbs   , numAabbs *        sizeof(AABB)           , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_materialsIdx     , materialsIdx , numObjects *      sizeof(int)            , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_sphereLights     , sphereLights , numSphereLights * sizeof(Float4)         , cudaMemcpyHostToDevice));

	// setup scene
	d_sceneGeometry.numSpheres      = numSpheres;
	d_sceneGeometry.spheres         = d_spheres;

	d_sceneGeometry.numAabbs        = numAabbs;
	d_sceneGeometry.aabbs           = d_sceneAabbs;

	d_sceneMaterial.numSphereLights = numSphereLights;
	d_sceneMaterial.sphereLights    = d_sphereLights;

	d_sceneMaterial.materials       = d_surfaceMaterials;
	d_sceneMaterial.materialsIdx    = d_materialsIdx;
	d_sceneMaterial.numMaterials    = numMaterials;

	d_sceneGeometry.triangles       = triangles;
	d_sceneGeometry.bvhNodes        = bvhNodes;
	d_sceneGeometry.numTriangles    = triCount;

	delete[] materials;
	delete[] materialsIdx;

	// cuda random
	h_randGen.init();
	d_randGen = h_randGen;

	// camera
	CameraSetup(cbo.camera);

	// textures
	texArrayUv = LoadTextureRgba8("resources/textures/colorChecker.png", sceneTextures.uv);
	//texArraySandAlbedo = LoadTextureRgb8("resources/textures/sand.png", sceneTextures.sandAlbedo);
	//texArraySandNormal = LoadTextureRgb8("resources/textures/sand_n.png", sceneTextures.sandNormal);

	// timer init
	timer.init();

	// set render dim to screen dim
	renderWidth = screenWidth;
	renderHeight = screenHeight;
}

void RayTracer::CameraSetup(Camera& camera)
{
	//cameraFocusPos = Float3(0, 1.0f, 0);
	//camera.pos = cameraFocusPos + Float3(7.3f, 2.0f, -6.9f);
	camera.pos = Float3(7.3f, 2.0f, -6.9f);

	//Float3 cameraLookAtPoint = cameraFocusPos;
	//Float3 camToObj = cameraLookAtPoint - camera.pos;

	//camera.dir = normalize(camToObj);
	camera.yaw = 0;
	camera.pitch = 0;
	camera.up  = { 0.0f, 1.0f, 0.0f };

	//camera.focal = camToObj.length();
	camera.focal = 5.0f;
	camera.aperture = 0.000001f;

	camera.resolution = { (float)renderWidth, (float)renderHeight };
	camera.fov.x = 90.0f * Pi_over_180;

	camera.update();
}

void RayTracer::cleanup()
{
	// ---------------- Destroy surface objects ----------------------
	// bvh
	cudaFree(constTriangles);
	cudaFree(triangles);
	cudaFree(sceneAabbs);
	cudaFree(sceneBoundingBox);
	cudaFree(morton);
	cudaFree(reorderIdx);
	cudaFree(bvhNodes);
	cudaFree(isAabbDone);

	// color buffer
    cudaDestroySurfaceObject(colorBufferA);
	cudaDestroySurfaceObject(colorBufferB);
	cudaFreeArray(colorBufferArrayA);
	cudaFreeArray(colorBufferArrayB);

	// down sized color buffer
	cudaDestroySurfaceObject(colorBuffer4);
	cudaDestroySurfaceObject(colorBuffer16);
	cudaDestroySurfaceObject(colorBuffer64);
	cudaFreeArray(colorBufferArray4);
	cudaFreeArray(colorBufferArray16);
	cudaFreeArray(colorBufferArray64);

	// bloom buffer
	cudaDestroySurfaceObject(bloomBuffer4);
	cudaDestroySurfaceObject(bloomBuffer16);
	cudaFreeArray(bloomBufferArray4);
	cudaFreeArray(bloomBufferArray16);

	// normal depth buffer
	cudaDestroySurfaceObject(normalDepthBufferA);
	cudaDestroySurfaceObject(normalDepthBufferB);
	cudaFreeArray(normalDepthBufferArrayA);
	cudaFreeArray(normalDepthBufferArrayB);

	// motion vector buffer
	cudaDestroySurfaceObject(motionVectorBuffer);
	cudaFreeArray(motionVectorBufferArray);

	// sample count buffer
	cudaDestroySurfaceObject(sampleCountBuffer);
	cudaFreeArray(sampleCountBufferArray);

	// noise level buffer
	cudaDestroySurfaceObject(noiseLevelBuffer);
	cudaFreeArray(noiseLevelBufferArray);

	// ---------------------- destroy texture objects --------------------------
	cudaDestroyTextureObject(sceneTextures.sandAlbedo);
	cudaDestroyTextureObject(sceneTextures.uv);
	cudaDestroyTextureObject(sceneTextures.sandNormal);
	if (texArraySandAlbedo != nullptr) cudaFreeArray(texArraySandAlbedo);
	if (texArrayUv         != nullptr) cudaFreeArray(texArrayUv);
	if (texArraySandNormal != nullptr) cudaFreeArray(texArraySandNormal);

	// sky
	cudaDestroyTextureObject(skyTex);
	cudaDestroySurfaceObject(skyBuffer);
	cudaFreeArray(skyArray);
	cudaFree(skyCdf);

	// --------------------- free other gpu buffer ----------------------------
	// exposure and histogram
	cudaFree(d_exposure);
	cudaFree(d_histogram);

	// scene
	cudaFree(d_spheres);
	cudaFree(d_surfaceMaterials);
	cudaFree(d_sceneAabbs);
	cudaFree(d_materialsIdx);
	cudaFree(d_sphereLights);

	// random
	h_randGen.clear();

	// free cpu buffer
	delete sceneAabbs;
	delete spheres;
	delete sphereLights;
}