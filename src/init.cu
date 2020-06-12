
#include "kernel.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>

inline float TrowbridgeReitzRoughnessToAlpha(float roughness)
{
    roughness = max1f(roughness, (float)1e-3);
    float x = std::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

void RayTracer::init(cudaStream_t* cudaStreams)
{
	streams = cudaStreams;

	// init terrain
	terrain.generateHeightMap();

	// AABB
	int numAabbs;
	AABB* aabbs = terrain.generateAabbs(numAabbs);

	// sphere
	numSpheres  = 6;
	spheres       = new Sphere[numSpheres];
	cameraFocusPos = Float3(0.0f, terrain.getHeightAt(0.0f) + 0.005f, 0.0f);
	spheres[0] = Sphere(cameraFocusPos, 0.005f);
	spheres[1] = Sphere(Float3(0.07f, terrain.getHeightAt(0.07f) + 0.005f, 0.0f) , 0.005f);
	spheres[2] = Sphere(Float3(-0.07f, terrain.getHeightAt(-0.07f) + 0.005f, 0.0f) , 0.005f);
	spheres[3] = Sphere(Float3(0.03f, terrain.getHeightAt(0.03f) - 0.005f, -0.01f) , 0.005f);
	spheres[4] = Sphere(Float3(-0.03f, terrain.getHeightAt(-0.03f) - 0.005f, -0.01f) , 0.005f);
	spheres[5] = Sphere(Float3(0.02f, terrain.getHeightAt(0.02f) + 0.005f, 0.0f) , 0.005f);

	// surface materials
	const int numMaterials     = 6;
	SurfaceMaterial* materials = new SurfaceMaterial[numMaterials];

	materials[0].type          = LAMBERTIAN_DIFFUSE;
	materials[0].albedo        = Float3(0.5f);

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
		//if (i < numSpheres + numAabbs / 2 + 10) materialsIdx[i] = 0;
		//else materialsIdx[i] = 4;
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

	// surface object
	cudaChannelFormatDesc surfaceChannelFormatDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	GpuErrorCheck(cudaMallocArray(&colorBufferArrayA, &surfaceChannelFormatDesc, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	GpuErrorCheck(cudaMallocArray(&colorBufferArrayB, &surfaceChannelFormatDesc, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	GpuErrorCheck(cudaMallocArray(&colorBufferArrayC, &surfaceChannelFormatDesc, renderWidth, renderHeight, cudaArraySurfaceLoadStore));

	bufferSize4 = UInt2(divRoundUp(renderWidth, 4u), divRoundUp(renderHeight, 4u));
	gridDim4 = dim3(divRoundUp(bufferSize4.x, blockDim.x), divRoundUp(bufferSize4.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray4, &surfaceChannelFormatDesc, bufferSize4.x, bufferSize4.y, cudaArraySurfaceLoadStore));
	GpuErrorCheck(cudaMallocArray(&bloomBufferArray4, &surfaceChannelFormatDesc, bufferSize4.x, bufferSize4.y, cudaArraySurfaceLoadStore));

	bufferSize16 = UInt2(divRoundUp(bufferSize4.x, 4u), divRoundUp(bufferSize4.y, 4u));
	gridDim16 = dim3(divRoundUp(bufferSize16.x, blockDim.x), divRoundUp(bufferSize16.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray16, &surfaceChannelFormatDesc, bufferSize16.x, bufferSize16.y, cudaArraySurfaceLoadStore));
	GpuErrorCheck(cudaMallocArray(&bloomBufferArray16, &surfaceChannelFormatDesc, bufferSize16.x, bufferSize16.y, cudaArraySurfaceLoadStore));

	bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u), divRoundUp(bufferSize16.y, 4u));
	gridDim64 = dim3(divRoundUp(bufferSize64.x, blockDim.x), divRoundUp(bufferSize64.y, blockDim.y), 1);
	GpuErrorCheck(cudaMallocArray(&colorBufferArray64, &surfaceChannelFormatDesc, bufferSize64.x, bufferSize64.y, cudaArraySurfaceLoadStore));

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;

	resDesc.res.array.array = colorBufferArrayA;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferA, &resDesc));
	resDesc.res.array.array = colorBufferArrayB;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferB, &resDesc));
	resDesc.res.array.array = colorBufferArrayC;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferC, &resDesc));

	resDesc.res.array.array = colorBufferArray4;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer4, &resDesc));
	resDesc.res.array.array = colorBufferArray16;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer16, &resDesc));
	resDesc.res.array.array = colorBufferArray64;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBuffer64, &resDesc));

	resDesc.res.array.array = bloomBufferArray4;
	GpuErrorCheck(cudaCreateSurfaceObject(&bloomBuffer4, &resDesc));
	resDesc.res.array.array = bloomBufferArray16;
	GpuErrorCheck(cudaCreateSurfaceObject(&bloomBuffer16, &resDesc));

	// uint buffer
	indexBuffers.init(renderBufferSize);

	GpuErrorCheck(cudaMalloc((void**)& gHitMask    , cbo.gridSize *      sizeof(ullint)));
	GpuErrorCheck(cudaMemset(gHitMask              , 0  , cbo.gridSize * sizeof(ullint)));

	GpuErrorCheck(cudaMalloc((void**)& d_exposure, 4 * sizeof(float)));
	GpuErrorCheck(cudaMalloc((void**)& d_histogram, 64 * sizeof(uint)));

	float initExposureLum[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GpuErrorCheck(cudaMemcpy(d_exposure, initExposureLum, 4 * sizeof(float), cudaMemcpyHostToDevice));

	// memory alloc
	GpuErrorCheck(cudaMalloc((void**)& d_spheres          , numSpheres *       sizeof(Sphere)));
	GpuErrorCheck(cudaMalloc((void**)& d_aabbs            , numAabbs *         sizeof(AABB)));
	GpuErrorCheck(cudaMalloc((void**)& d_materialsIdx     , numObjects *       sizeof(int)));
	GpuErrorCheck(cudaMalloc((void**)& d_surfaceMaterials , numMaterials *     sizeof(SurfaceMaterial)));
	GpuErrorCheck(cudaMalloc((void**)& d_sphereLights     , numSphereLights *  sizeof(Float4)));

	GpuErrorCheck(cudaMalloc((void**)& rayStates          , renderBufferSize * sizeof(RayState)));

	// memory copy
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

	// camera
	Camera& camera = cbo.camera;

	camera.resolution = Float4((float)renderWidth, (float)renderHeight, 1.0f / renderWidth, 1.0f / renderHeight);

	camera.fov.x = 90.0f * Pi_over_180;
	camera.fov.y = camera.fov.x * (float)renderHeight / (float)renderWidth;
	camera.fov.z = tan(camera.fov.x / 2.0f);
	camera.fov.w = tan(camera.fov.y / 2.0f);

	Float3 cameraLookAtPoint = cameraFocusPos + Float3(0.0f, 0.01f, 0.0f);
	camera.pos               = cameraFocusPos + Float3(0.0f, 0.0f, -0.1f);
	camera.dir               = normalize(cameraLookAtPoint - camera.pos.xyz);
	camera.left              = cross(Float3(0, 1, 0), camera.dir.xyz);

	timer.init();

	LoadTextures();
}

void RayTracer::LoadTextures()
{
	{
		int texWidth, texHeight, texChannel;
		const char* texPath = "resources/textures/uv.png";
		unsigned char* buffer = stbi_load(texPath, &texWidth, &texHeight, &texChannel, STBI_rgb_alpha);
		std::cout << "Texture loaded: " << texPath << ", width=" << texWidth << ", height=" << texHeight << ", channel=" << texChannel << "\n";
		assert(buffer != NULL);

		uchar4* cpuTextureBuffer = new uchar4[texWidth * texHeight];
		for (int i = 0; i < texWidth * texHeight; ++i)
		{
			cpuTextureBuffer[i] = make_uchar4(buffer[i * 4], buffer[i * 4 + 1], buffer[i * 4 + 2], buffer[i * 4 + 3]);
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		cudaMallocArray(&texArrayUv, &channelDesc, texWidth, texHeight);
		cudaMemcpyToArray(texArrayUv, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(uchar4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = texArrayUv;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0]   = cudaAddressModeWrap;
		texDesc.addressMode[1]   = cudaAddressModeWrap;
		texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.readMode         = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&sceneTextures.uv, &resDesc, &texDesc, NULL);
		assert(sceneTextures.uv != NULL);

		delete cpuTextureBuffer;
		stbi_image_free(buffer);
	}

#if 0
	{
		int texWidth, texHeight, texChannel;
		const char* texPath = "resources/textures/TexturesCom_DesertSand2_3x3_1K_albedo.png";
		unsigned char* buffer = stbi_load(texPath, &texWidth, &texHeight, &texChannel, STBI_rgb);
		std::cout << "Texture loaded: " << texPath << ", width=" << texWidth << ", height=" << texHeight << ", channel=" << texChannel << "\n";
		assert(buffer != NULL);

		uchar4* cpuTextureBuffer = new uchar4[texWidth * texHeight];
		for (int i = 0; i < texWidth * texHeight; ++i)
		{
			cpuTextureBuffer[i] = make_uchar4(buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2], 0);
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		cudaMallocArray(&texArraySandAlbedo, &channelDesc, texWidth, texHeight);
		cudaMemcpyToArray(texArraySandAlbedo, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(uchar4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = texArraySandAlbedo;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0]   = cudaAddressModeWrap;
		texDesc.addressMode[1]   = cudaAddressModeWrap;
		texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.readMode         = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&sceneTextures.sandAlbedo, &resDesc, &texDesc, NULL);
		assert(sceneTextures.sandAlbedo != NULL);

		delete cpuTextureBuffer;
		stbi_image_free(buffer);
	}

	{
		int texWidth, texHeight, texChannel;
		const char* texPath = "resources/textures/TexturesCom_DesertSand2_3x3_1K_normal.png";
		unsigned short* buffer = stbi_load_16(texPath, &texWidth, &texHeight, &texChannel, STBI_rgb);
		std::cout << "Texture loaded: " << texPath << ", width=" << texWidth << ", height=" << texHeight << ", channel=" << texChannel << "\n";
		assert(buffer != NULL);

		ushort4* cpuTextureBuffer = new ushort4[texWidth * texHeight];
		for (int i = 0; i < texWidth * texHeight; ++i)
		{
			cpuTextureBuffer[i] = make_ushort4(buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2], 0);
		}

		cudaArray* &texArray = texArraySandNormal;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<ushort4>();
		cudaMallocArray(&texArray, &channelDesc, texWidth, texHeight);
		cudaMemcpyToArray(texArray, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(ushort4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = texArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0]   = cudaAddressModeWrap;
		texDesc.addressMode[1]   = cudaAddressModeWrap;
		texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.readMode         = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&sceneTextures.sandNormal, &resDesc, &texDesc, NULL);
		assert(sceneTextures.sandNormal != NULL);

		delete cpuTextureBuffer;
		stbi_image_free(buffer);
	}
#endif

	{
		int texWidth, texHeight, texChannel;
		const char* texPath = "resources/textures/sand.png";
		unsigned char* buffer = stbi_load(texPath, &texWidth, &texHeight, &texChannel, STBI_rgb);
		std::cout << "Texture loaded: " << texPath << ", width=" << texWidth << ", height=" << texHeight << ", channel=" << texChannel << "\n";
		assert(buffer != NULL);

		uchar4* cpuTextureBuffer = new uchar4[texWidth * texHeight];
		for (int i = 0; i < texWidth * texHeight; ++i)
		{
			cpuTextureBuffer[i] = make_uchar4(buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2], 0);
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		cudaMallocArray(&texArraySandAlbedo, &channelDesc, texWidth, texHeight);
		cudaMemcpyToArray(texArraySandAlbedo, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(uchar4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = texArraySandAlbedo;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0]   = cudaAddressModeWrap;
		texDesc.addressMode[1]   = cudaAddressModeWrap;
		texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.readMode         = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&sceneTextures.sandAlbedo, &resDesc, &texDesc, NULL);
		assert(sceneTextures.sandAlbedo != NULL);

		delete cpuTextureBuffer;
		stbi_image_free(buffer);
	}

	{
		int texWidth, texHeight, texChannel;
		const char* texPath = "resources/textures/sand_n.png";
		unsigned char* buffer = stbi_load(texPath, &texWidth, &texHeight, &texChannel, STBI_rgb);
		std::cout << "Texture loaded: " << texPath << ", width=" << texWidth << ", height=" << texHeight << ", channel=" << texChannel << "\n";
		assert(buffer != NULL);

		uchar4* cpuTextureBuffer = new uchar4[texWidth * texHeight];
		for (int i = 0; i < texWidth * texHeight; ++i)
		{
			cpuTextureBuffer[i] = make_uchar4(buffer[i * 3], buffer[i * 3 + 1], buffer[i * 3 + 2], 0);
		}

		cudaArray* &texArray = texArraySandNormal;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		cudaMallocArray(&texArray, &channelDesc, texWidth, texHeight);
		cudaMemcpyToArray(texArray, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(uchar4), cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = texArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0]   = cudaAddressModeWrap;
		texDesc.addressMode[1]   = cudaAddressModeWrap;
		texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.readMode         = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&sceneTextures.sandNormal, &resDesc, &texDesc, NULL);
		assert(sceneTextures.sandNormal != NULL);

		delete cpuTextureBuffer;
		stbi_image_free(buffer);
	}
}

void RayTracer::cleanup()
{
	delete[] spheres;
	delete[] sphereLights;

	// Destroy surface objects
    cudaDestroySurfaceObject(colorBufferA);
	cudaDestroySurfaceObject(colorBufferB);
	cudaDestroySurfaceObject(colorBufferC);

	cudaDestroySurfaceObject(colorBuffer4);
	cudaDestroySurfaceObject(colorBuffer16);
	cudaDestroySurfaceObject(colorBuffer64);

    // Free device memory
    cudaFreeArray(colorBufferArrayA);
	cudaFreeArray(colorBufferArrayB);
	cudaFreeArray(colorBufferArrayC);

	cudaFreeArray(colorBufferArray4);
	cudaFreeArray(colorBufferArray16);
	cudaFreeArray(colorBufferArray64);
	indexBuffers.cleanUp();

	cudaDestroyTextureObject(sceneTextures.sandAlbedo);
	cudaFreeArray(texArraySandAlbedo);

	cudaDestroyTextureObject(sceneTextures.uv);
	cudaFreeArray(texArrayUv);

	cudaDestroyTextureObject(sceneTextures.sandNormal);
	cudaFreeArray(texArraySandNormal);

	cudaFree(d_exposure);
	cudaFree(d_histogram);

	cudaFree(gHitMask);

	cudaFree(d_spheres);
	cudaFree(d_surfaceMaterials);
	cudaFree(d_aabbs);
	cudaFree(d_materialsIdx);
	cudaFree(d_sphereLights);

	cudaFree(d_randInitVec);

	cudaFree(rayStates);
}