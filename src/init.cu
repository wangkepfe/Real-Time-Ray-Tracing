
#include "kernel.cuh"
#include "textureUtils.cuh"
#include "blueNoiseRandGenData.h"

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
	//terrain.generateHeightMap();

	// AABB
	//int numAabbs;
	//AABB* aabbs = terrain.generateAabbs(numAabbs);
	int numAabbs = 3;
	AABB* aabbs = new AABB[numAabbs];
	aabbs[0] = AABB({-0.05f, 0.005f, 0.0f}, 0.01f);
	aabbs[1] = AABB({-0.03f, 0.005f, 0.0f}, 0.01f);
	aabbs[2] = AABB({0.05f, 0.005f, 0.0f}, 0.01f);

	// sphere
	numSpheres = 3;
	spheres    = new Sphere[numSpheres];
	spheres[0] = Sphere({-0.05f, 0.015f, 0.0f}, 0.005f);
	spheres[1] = Sphere({-0.03f, 0.015f, 0.0f}, 0.005f);
	spheres[2] = Sphere({0.05f, 0.015f, 0.0f}, 0.005f);

	// triangles
	numTriangles   = 2;
	triangles = new Triangle[numTriangles];
	triangles[0] = Triangle({0.0f, 0.0f, 0.0f}, {0.02f, 0.0f, 0.0f}, {0.02f, 0.02f, 0.0f});
	triangles[1] = Triangle({0.0f, 0.0f, 0.0f}, {0.02f, 0.02f, 0.0f}, {0.0f, 0.02f, 0.0f});
	//for (int i = 0 ; i < numTriangles; ++i) { triangles[i].WatertightTransform(); }

	// surface materials
	const int numMaterials     = 6;
	SurfaceMaterial* materials = new SurfaceMaterial[numMaterials];

	materials[0].type          = EMISSIVE;
	materials[0].albedo        = Float3(0.1f, 0.2f, 0.9f);

	materials[1].type          = PERFECT_FRESNEL_REFLECTION_REFRACTION;

	materials[2].type          = EMISSIVE;
	materials[2].albedo        = Float3(0.9f, 0.2f, 0.1f);

	materials[3].type          = LAMBERTIAN_DIFFUSE;
	materials[3].albedo        = Float3(0.5f);

	materials[4].type          = PERFECT_REFLECTION;

	materials[5].type          = MICROFACET_REFLECTION;
	materials[5].albedo        = Float3(0.5f);
	materials[5].alpha         = TrowbridgeReitzRoughnessToAlpha(0.01f);

	// material index for each object
	const int numObjects = numAabbs + numSpheres;
	int* materialsIdx = new int[numObjects];

	// sphere
	materialsIdx[0] = 0;
	materialsIdx[1] = 1;
	materialsIdx[2] = 2;

	// aabb
	materialsIdx[3] = 3;
	materialsIdx[4] = 4;
	materialsIdx[5] = 5;

	// triangle
	materialsIdx[6] = 5;
	materialsIdx[7] = 5;

	// light source
	numSphereLights = 2;
	sphereLights = new Sphere[numSphereLights];
	sphereLights[0] = spheres[0];
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
	GpuErrorCheck(cudaMalloc((void**)& d_triangles        , numTriangles *     sizeof(Triangle)));

	GpuErrorCheck(cudaMemcpy(d_spheres          , spheres      , numSpheres *      sizeof(Float4)         , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_surfaceMaterials , materials    , numMaterials *    sizeof(SurfaceMaterial), cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_aabbs            , aabbs        , numAabbs *        sizeof(AABB)           , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_materialsIdx     , materialsIdx , numObjects *      sizeof(int)            , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_sphereLights     , sphereLights , numSphereLights * sizeof(Float4)         , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_triangles        , triangles    , numTriangles *    sizeof(Triangle)       , cudaMemcpyHostToDevice));

	// setup scene
	d_sceneGeometry.numSpheres      = numSpheres;
	d_sceneGeometry.spheres         = d_spheres;

	d_sceneGeometry.numAabbs        = numAabbs;
	d_sceneGeometry.aabbs           = d_aabbs;

	d_sceneGeometry.numTriangles    = numTriangles;
	d_sceneGeometry.triangles       = d_triangles;

	d_sceneMaterial.numSphereLights = numSphereLights;
	d_sceneMaterial.sphereLights    = d_sphereLights;

	d_sceneMaterial.materials       = d_surfaceMaterials;
	d_sceneMaterial.materialsIdx    = d_materialsIdx;
	d_sceneMaterial.numMaterials    = numMaterials;

	delete[] materials;
	delete[] materialsIdx;

	// cuda random
	h_randGen.init();
	d_randGen = h_randGen;

	// camera
	cameraFocusPos = {0, 0, 0};
	Camera& camera = cbo.camera;
	CameraSetup(camera);

	// textures
	LoadTextureRgba8("resources/textures/uv.png", texArrayUv, sceneTextures.uv);
	LoadTextureRgb8("resources/textures/sand.png", texArraySandAlbedo, sceneTextures.sandAlbedo);
	LoadTextureRgb8("resources/textures/sand_n.png", texArraySandNormal, sceneTextures.sandNormal);

	// timer init
	timer.init();
}

void RayTracer::CameraSetup(Camera& camera)
{
	camera.pos = cameraFocusPos + Float3(0.0f, 0.01f, -0.1f);

	Float3 cameraLookAtPoint = cameraFocusPos;
	Float3 camToObj = cameraLookAtPoint - camera.pos;

	camera.dir = normalize(camToObj);
	camera.up  = { 0.0f, 1.0f, 0.0f };

	camera.focal = camToObj.length();
	camera.aperture = 0.0001f;

	camera.resolution = { (float)renderWidth, (float)renderHeight };
	camera.fov.x = 90.0f * Pi_over_180;

	camera.update();
}

void RayTracer::cleanup()
{
	// free cpu buffer
	delete[] aabbs;
	delete[] spheres;
	delete[] sphereLights;
	delete[] triangles;

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
	h_randGen.clear();
}