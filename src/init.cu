
#include "kernel.cuh"
#include "textureUtils.cuh"

inline float TrowbridgeReitzRoughnessToAlpha(float roughness)
{
    roughness = max1f(roughness, (float)1e-3);
    float x = std::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

void RayTracer::init(cudaStream_t* cudaStreams)
{
	// set streams
	streams = cudaStreams;

	// init terrain
	terrain.generateHeightMap();

	// AABB
	int numAabbs;
	AABB* aabbs = terrain.generateAabbs(numAabbs);

	// sphere
	numSpheres     = 6;
	spheres        = new Sphere[numSpheres];
	cameraFocusPos = Float3(0.0f, terrain.getHeightAt(0.0f) + 0.005f, 0.0f);
	spheres[0]     = Sphere(cameraFocusPos, 0.005f);
	spheres[1]     = Sphere(Float3(0.07f, terrain.getHeightAt(0.07f) + 0.005f, 0.0f) , 0.005f);
	spheres[2]     = Sphere(Float3(-0.07f, terrain.getHeightAt(-0.07f) + 0.005f, 0.0f) , 0.005f);
	spheres[3]     = Sphere(Float3(0.03f, terrain.getHeightAt(0.03f) - 0.005f, -0.01f) , 0.005f);
	spheres[4]     = Sphere(Float3(-0.03f, terrain.getHeightAt(-0.03f) - 0.005f, -0.01f) , 0.005f);
	spheres[5]     = Sphere(Float3(0.02f, terrain.getHeightAt(0.02f) + 0.005f, 0.0f) , 0.005f);

	// surface materials
	const int numMaterials     = 6;
	SurfaceMaterial* materials = new SurfaceMaterial[numMaterials];

	materials[0].type          = MICROFACET_REFLECTION;
	materials[0].albedo        = Float3(0.5f);
	materials[0].alpha         = TrowbridgeReitzRoughnessToAlpha(0.9f);
	materials[0].F0            = Float3(0.05f);

	materials[1].type          = MICROFACET_REFLECTION;
	materials[1].albedo        = Float3(0.5f);
	materials[1].alpha         = TrowbridgeReitzRoughnessToAlpha(0.01f);

	materials[2].type          = EMISSIVE;
	materials[2].albedo        = Float3(0.9f, 0.2f, 0.1f);

	materials[3].type          = EMISSIVE;
	materials[3].albedo        = Float3(0.1f, 0.2f, 0.9f);

	materials[4].type          = MICROFACET_REFLECTION;
	materials[4].albedo        = Float3(0.5f, 0.5f, 0.1f);
	materials[4].alpha         = TrowbridgeReitzRoughnessToAlpha(0.5f);
	materials[4].F0            = Float3(0.05f);

	materials[5].type          = PERFECT_FRESNEL_REFLECTION_REFRACTION;

	// material index for each object
	const int numObjects = numAabbs + numSpheres;
	int* materialsIdx    = new int[numObjects];
	materialsIdx[0] = 1;
	materialsIdx[1] = 2;
	materialsIdx[2] = 3;
	materialsIdx[3] = 4;
	materialsIdx[4] = 5;
	materialsIdx[5] = 5;
	for (int i = numSpheres; i < numObjects; ++i)
	{
		materialsIdx[i] = 0;
	}

	// light source
	numSphereLights = 2;
	sphereLights = new Sphere[numSphereLights];
	sphereLights[0] = spheres[1];
	sphereLights[1] = spheres[2];

	// init cuda
	gpuDeviceInit(0);

	// constant buffer
	cbo.maxSceneLoop = 4;
	cbo.frameNum = 0;
	cbo.aaSampleNum = 8;
	cbo.bufferSize = renderBufferSize;
	cbo.bufferDim = UInt2(renderWidth, renderHeight);

	// launch param
	blockDim = dim3(8, 8, 1);
	gridDim = dim3(divRoundUp(renderWidth, blockDim.x), divRoundUp(renderHeight, blockDim.y), 1);

	cbo.gridDim = UInt2(gridDim.x, gridDim.y);
	cbo.gridSize = gridDim.x * gridDim.y;

	scaleBlockDim = dim3(8, 8, 1);
	scaleGridDim = dim3(divRoundUp(screenWidth, scaleBlockDim.x), divRoundUp(screenHeight, scaleBlockDim.y), 1);

	// ------------------------ surface/texture object ---------------------------
	cudaChannelFormatDesc channelFormatRgba32 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelFormatRgba16 = cudaCreateChannelDescHalf4();

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
	GpuErrorCheck(cudaMallocArray(&colorBufferArrayA, &channelFormatRgba32, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArrayA;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferA, &resDesc));

	// array B: TAA buffer
	GpuErrorCheck(cudaMallocArray(&colorBufferArrayB, &channelFormatRgba32, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArrayB;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferB, &resDesc));

	// color buffer 1/4 size
	bufferSize4 = UInt2(divRoundUp(renderWidth, 4u), divRoundUp(renderHeight, 4u));
	gridDim4 = dim3(divRoundUp(bufferSize4.x, blockDim.x), divRoundUp(bufferSize4.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray4, &channelFormatRgba16, bufferSize4.x, bufferSize4.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArray4;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer4, &resDesc));

	// bloom buffer 1/4 size
	GpuErrorCheck(cudaMallocArray(&bloomBufferArray4, &channelFormatRgba16, bufferSize4.x, bufferSize4.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = bloomBufferArray4;
	GpuErrorCheck(cudaCreateSurfaceObject(&bloomBuffer4, &resDesc));

	// color buffer 1/16 size
	bufferSize16 = UInt2(divRoundUp(bufferSize4.x, 4u), divRoundUp(bufferSize4.y, 4u));
	gridDim16 = dim3(divRoundUp(bufferSize16.x, blockDim.x), divRoundUp(bufferSize16.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray16, &channelFormatRgba16, bufferSize16.x, bufferSize16.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArray16;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer16, &resDesc));

	// bloom buffer 1/16 size
	GpuErrorCheck(cudaMallocArray(&bloomBufferArray16, &channelFormatRgba16, bufferSize16.x, bufferSize16.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = bloomBufferArray16;
	GpuErrorCheck(cudaCreateSurfaceObject(&bloomBuffer16, &resDesc));

	// color buffer 1/64 size
	bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u), divRoundUp(bufferSize16.y, 4u));
	gridDim64 = dim3(divRoundUp(bufferSize64.x, blockDim.x), divRoundUp(bufferSize64.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray64, &channelFormatRgba16, bufferSize64.x, bufferSize64.y, cudaArraySurfaceLoadStore));
	resDesc.res.array.array = colorBufferArray64;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer64, &resDesc));

	// ----------------------- sky buffer ------------------------
	GpuErrorCheck(cudaMallocArray(&skyArray, &channelFormatRgba32, skyWidth, skyHeight, cudaArraySurfaceLoadStore));
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
	GpuErrorCheck(cudaMalloc((void**)& d_aabbs            , numAabbs *         sizeof(AABB)));
	GpuErrorCheck(cudaMalloc((void**)& d_materialsIdx     , numObjects *       sizeof(int)));
	GpuErrorCheck(cudaMalloc((void**)& d_surfaceMaterials , numMaterials *     sizeof(SurfaceMaterial)));
	GpuErrorCheck(cudaMalloc((void**)& d_sphereLights     , numSphereLights *  sizeof(Float4)));

	GpuErrorCheck(cudaMemcpy(d_spheres          , spheres      , numSpheres *      sizeof(Float4)         , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_surfaceMaterials , materials    , numMaterials *    sizeof(SurfaceMaterial), cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_aabbs            , aabbs        , numAabbs *        sizeof(AABB)           , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_materialsIdx     , materialsIdx , numObjects *      sizeof(int)            , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_sphereLights     , sphereLights , numSphereLights * sizeof(Float4)         , cudaMemcpyHostToDevice));

	// setup scene
	d_sceneGeometry.numSpheres      = numSpheres;
	d_sceneGeometry.spheres         = d_spheres;

	d_sceneGeometry.numAabbs        = numAabbs;
	d_sceneGeometry.aabbs           = d_aabbs;

	d_sceneMaterial.numSphereLights = numSphereLights;
	d_sceneMaterial.sphereLights    = d_sphereLights;

	d_sceneMaterial.materials       = d_surfaceMaterials;
	d_sceneMaterial.materialsIdx    = d_materialsIdx;
	d_sceneMaterial.numMaterials    = numMaterials;

	delete[] materials;
	delete[] materialsIdx;

	// cuda random
	GpuErrorCheck(cudaMalloc((void**)& d_randInitVec, 3 * sizeof(RandInitVec)));
	RandInitVec* randDirvectors = g_curandDirectionVectors32;
	GpuErrorCheck(cudaMemcpy(d_randInitVec, randDirvectors, 3 * sizeof(RandInitVec), cudaMemcpyHostToDevice));

	// --------------------------------- camera ------------------------------------
	Camera& camera = cbo.camera;

	camera.resolution = Float4((float)renderWidth, (float)renderHeight, 1.0f / renderWidth, 1.0f / renderHeight);

	camera.fov.x = 90.0f * Pi_over_180;
	camera.fov.y = camera.fov.x * (float)renderHeight / (float)renderWidth;
	camera.fov.z = tan(camera.fov.x / 2.0f);
	camera.fov.w = tan(camera.fov.y / 2.0f);

	Float3 cameraLookAtPoint = cameraFocusPos + Float3(0.0f, 0.01f, 0.0f);
	camera.pos               = cameraFocusPos + Float3(0.0f, 0.0f, -0.1f);
	Float3 CamToObj          = cameraLookAtPoint - camera.pos.xyz;
	camera.dirFocal.xyz      = normalize(CamToObj);
	camera.dirFocal.w        = CamToObj.length();
	camera.leftAperture.xyz  = normalize(cross(Float3(0, 1, 0), camera.dirFocal.xyz));
	camera.leftAperture.w    = 0.002f;
	camera.up.xyz            = normalize(cross(camera.dirFocal.xyz, camera.leftAperture.xyz));

	// textures
	LoadTextureRgba8("resources/textures/uv.png", texArrayUv, sceneTextures.uv);
	LoadTextureRgb8("resources/textures/sand.png", texArraySandAlbedo, sceneTextures.sandAlbedo);
	LoadTextureRgb8("resources/textures/sand_n.png", texArraySandNormal, sceneTextures.sandNormal);

	// timer init
	timer.init();
}

void RayTracer::cleanup()
{
	// free cpu buffer
	delete[] spheres;
	delete[] sphereLights;

	// ---------------- Destroy surface objects ----------------------
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

	// ---------------------- destroy texture objects --------------------------
	cudaDestroyTextureObject(sceneTextures.sandAlbedo);
	cudaDestroyTextureObject(sceneTextures.uv);
	cudaDestroyTextureObject(sceneTextures.sandNormal);
	cudaFreeArray(texArraySandAlbedo);
	cudaFreeArray(texArrayUv);
	cudaFreeArray(texArraySandNormal);

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
	cudaFree(d_aabbs);
	cudaFree(d_materialsIdx);
	cudaFree(d_sphereLights);

	// random
	cudaFree(d_randInitVec);
}