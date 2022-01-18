
#include "kernel.cuh"
#include "fileUtils.cuh"
#include "blueNoiseRandGenData.h"
#include "cuda_fp16.h"
#include "terrain.h"

extern GlobalSettings* g_settings;

template<typename T>
__global__ void InitBuffer(T val, SurfObj buffer, Int2 bufferSize)
{
	Int2 idx;
	idx.x = blockIdx.x * blockDim.x + threadIdx.x;
	idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= bufferSize.x || idx.y >= bufferSize.y) return;

	surf2Dwrite(val, buffer, idx.x, idx.y, cudaBoundaryModeClamp);
}

namespace
{
void LoadTrianglesFromFile(std::vector<Triangle>& h_triangles, uint& triCount)
{
	std::string fileName = g_settings->inputMeshFileName;
	std::ifstream infile (fileName, std::ifstream::binary);
	if (infile.good())
	{
		size_t currentSize = sizeof(uint);
		char* pTriCount = new char[currentSize];
		infile.read(pTriCount, currentSize);
		triCount = *reinterpret_cast<uint*>(pTriCount);

		currentSize = sizeof(Triangle) * triCount;
		char* pTrianglesRaw = new char[currentSize];
		infile.read(pTrianglesRaw, currentSize);
		Triangle* pTriangles = reinterpret_cast<Triangle*>(pTrianglesRaw);
		h_triangles.assign(pTriangles, pTriangles + triCount);

		infile.close();
		std::cout << "Successfully read scene data from \"" << fileName << "\"!\n";
	} else {
		std::cout << "Error: Failed to read scene data from \"" << fileName << "\".\n";
	}
}
} // namespace

void RayTracer::init(cudaStream_t* cudaStreams)
{
	maxRenderWidth = g_settings->maxWidth;
	maxRenderHeight = g_settings->maxHeight;

	if (g_settings->useDynamicResolution)
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

	{
		// load triangles
		std::vector<Triangle> h_triangles;

		//LoadTrianglesFromFile(h_triangles, triCount);

		pTerrainGenerator = new TerrainGenerator(h_triangles);
		pTerrainGenerator->Generate();
		triCount = h_triangles.size();

		// pad with repeat triangles, required by update geometry
		triCountPadded = triCount;
		if (triCountPadded % KernalBatchSize != 0)
		{
			triCountPadded += (KernalBatchSize - triCount % KernalBatchSize);
			h_triangles.resize(triCountPadded);
			for (int i = triCount; i < triCountPadded; ++i)
			{
				h_triangles[i] = h_triangles[0];
			}
		}

		// pad size for radix sort
		triCountPadded2 = triCount;
		if (triCountPadded2 % BatchSize != 0)
		{
			triCountPadded2 += (BatchSize - triCount % BatchSize);
		}

		// batch count
		batchCount = triCountPadded2 / BatchSize;
		assert(batchCount < BatchSize);

		// triangle batch count array
		std::vector<uint> h_triCountArray(batchCount, BatchSize);
		h_triCountArray[batchCount - 1] = triCount - (triCountPadded2 - BatchSize);

		// copy triangle data to gpu
		const uint allocSize = triCountPadded * sizeof(Triangle);
		GpuErrorCheck(cudaMalloc((void**)& constTriangles, triCountPadded * sizeof(Triangle)));
		GpuErrorCheck(cudaMemcpy(constTriangles, h_triangles.data(), triCountPadded * sizeof(Triangle), cudaMemcpyHostToDevice));

		// copy batch count to gpu
		GpuErrorCheck(cudaMalloc((void**)& batchCountArray, 1 * sizeof(uint)));
		GpuErrorCheck(cudaMemcpy(batchCountArray, &batchCount, 1 * sizeof(uint), cudaMemcpyHostToDevice));

		// copy tri count of each batch to gpu
		GpuErrorCheck(cudaMalloc((void**)& triCountArray, batchCount * sizeof(uint)));
		GpuErrorCheck(cudaMemcpy(triCountArray, h_triCountArray.data(), batchCount * sizeof(uint), cudaMemcpyHostToDevice));
	}

	// -------------------------------- bvh ---------------------------------------
	// triangle
	GpuErrorCheck(cudaMalloc((void**)& triangles, triCountPadded * sizeof(Triangle)));

	// aabb
	GpuErrorCheck(cudaMalloc((void**)& aabbs, triCountPadded * sizeof(AABB)));

	// morton code
	GpuErrorCheck(cudaMalloc((void**)& morton, triCountPadded2 * sizeof(uint)));
	GpuErrorCheck(cudaMemset(morton, UINT_MAX, triCountPadded2 * sizeof(uint))); // init morton code to UINT_MAX

	// reorder idx
	GpuErrorCheck(cudaMalloc((void**)& reorderIdx, triCountPadded2 * sizeof(uint)));

	// bvh nodes
	GpuErrorCheck(cudaMalloc((void**)& bvhNodes, triCountPadded * sizeof(BVHNode)));

	//------------------------------------ tlas -------------------------------------------

	// aabb
	GpuErrorCheck(cudaMalloc((void**)& tlasAabbs, batchCount * sizeof(AABB)));

	// morton code
	GpuErrorCheck(cudaMalloc((void**)& tlasMorton, BatchSize * sizeof(uint)));
	GpuErrorCheck(cudaMemset(tlasMorton, UINT_MAX, BatchSize * sizeof(uint))); // init morton code to UINT_MAX

	// reorder idx
	GpuErrorCheck(cudaMalloc((void**)& tlasReorderIdx, BatchSize * sizeof(uint)));

	// bvh nodes
	GpuErrorCheck(cudaMalloc((void**)& tlasBvhNodes, batchCount * sizeof(BVHNode)));

	//-------------------------------------------------------------------------------

	// AABB
	int numAabbs = 2;
	sceneAabbs = new AABB[numAabbs];
	i = 0;
	sceneAabbs[i++] = AABB({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
	sceneAabbs[i++] = AABB({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});

	// sphere
	#if RENDER_SPHERE
	numSpheres = 2;
	spheres    = new Sphere[numSpheres];
	i = 0;
	spheres[i++] = Sphere({0.0f, 1.0f, 4.0f}, 1.0f);
	spheres[i++] = Sphere({0.0f, 1.0f, -4.0f}, 1.0f);
	#endif

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
	materials[i].type          = PERFECT_REFLECTION;
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
	int numObjects = triCount;

	#if RENDER_SPHERE
	numObjects += numSpheres;
	#endif

	// material index
	int* materialsIdx = new int[numObjects];
	for (i = 0; i < triCount; ++i)
	{
		materialsIdx[i] = 3;
	}
	#if RENDER_SPHERE
	materialsIdx[i++] = 0;
	materialsIdx[i++] = 2;
	#endif

	// light source
	#if RENDER_SPHERE_LIGHT
	numSphereLights = 2;
	sphereLights = new Sphere[numSphereLights];
	for (i = 0; i < numSphereLights; ++i)
	{
		sphereLights[i] = spheres[i];
	}
	#endif

	// constant buffer
	cbo.frameNum = 0;
	cbo.bvhDebugLevel = -1;
	cbo.bvhBatchSize = BatchSize;

	// launch param
	blockDim = dim3(8, 8, 1);
	gridDim = dim3(divRoundUp(renderWidth, blockDim.x), divRoundUp(renderHeight, blockDim.y), 1);

	scaleBlockDim = dim3(8, 8, 1);
	scaleGridDim = dim3(divRoundUp(screenWidth, scaleBlockDim.x), divRoundUp(screenHeight, scaleBlockDim.y), 1);

	bufferSize4 = UInt2(divRoundUp(renderWidth, 4u), divRoundUp(renderHeight, 4u));
	gridDim4 = dim3(divRoundUp(bufferSize4.x, blockDim.x), divRoundUp(bufferSize4.y, blockDim.y), 1);

	bufferSize16 = UInt2(divRoundUp(bufferSize4.x, 4u), divRoundUp(bufferSize4.y, 4u));
	gridDim16 = dim3(divRoundUp(bufferSize16.x, blockDim.x), divRoundUp(bufferSize16.y, blockDim.y), 1);

	bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u), divRoundUp(bufferSize16.y, 4u));
	gridDim64 = dim3(divRoundUp(bufferSize64.x, blockDim.x), divRoundUp(bufferSize64.y, blockDim.y), 1);

	// texture description
	// cudaTextureDesc texDesc  = {};
	// texDesc.addressMode[0]   = cudaAddressModeClamp;
	// texDesc.addressMode[1]   = cudaAddressModeClamp;
	// texDesc.filterMode       = cudaFilterModeLinear;
	// texDesc.readMode         = cudaReadModeElementType;
	// texDesc.normalizedCoords = 1;

	buffer2DManager.init(renderWidth, renderHeight, screenWidth, screenHeight);

	//InitBuffer<<<dim3(divRoundUp(gridDim.x, 8), divRoundUp(gridDim.y, 8), 1), dim3(8, 8, 1)>>> (make_ushort1(0), GetBuffer2D(NoiseLevelBuffer), Int2(gridDim.x, gridDim.y));

	GpuErrorCheck(cudaMalloc((void**)&skyCdf, skySize * sizeof(float)));
	GpuErrorCheck(cudaMemset(skyCdf, 0, skySize * sizeof(float)));

	// exposure
	GpuErrorCheck(cudaMalloc((void**)& d_exposure, 4 * sizeof(float)));
	float initExposureLum[4] = { 1.0f, 1.0f, 1.0f, 1.0f }; // (exposureValue, historyAverageLuminance, historyBrightThresholdLuminance, unused)
	GpuErrorCheck(cudaMemcpy(d_exposure, initExposureLum, 4 * sizeof(float), cudaMemcpyHostToDevice));

	// histogram
	GpuErrorCheck(cudaMalloc((void**)& d_histogram, 64 * sizeof(uint)));

	// for debug
	GpuErrorCheck(cudaMalloc((void**)& dumpFrameBuffer, screenWidth * screenHeight * sizeof(uchar4)));
	GpuErrorCheck(cudaMemset(dumpFrameBuffer, 0, screenWidth * screenHeight * sizeof(uchar4)));

	// scene
	GpuErrorCheck(cudaMalloc((void**)& d_sceneAabbs       , numAabbs *         sizeof(AABB)));
	GpuErrorCheck(cudaMalloc((void**)& d_materialsIdx     , numObjects *       sizeof(int)));
	GpuErrorCheck(cudaMalloc((void**)& d_surfaceMaterials , numMaterials *     sizeof(SurfaceMaterial)));
	#if RENDER_SPHERE
	GpuErrorCheck(cudaMalloc((void**)& d_spheres          , numSpheres *       sizeof(Sphere)));
	#endif
	#if RENDER_SPHERE_LIGHT
	GpuErrorCheck(cudaMalloc((void**)& d_sphereLights     , numSphereLights *  sizeof(Float4)));
	#endif

	
	GpuErrorCheck(cudaMemcpy(d_surfaceMaterials , materials    , numMaterials *    sizeof(SurfaceMaterial), cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_sceneAabbs       , sceneAabbs   , numAabbs *        sizeof(AABB)           , cudaMemcpyHostToDevice));
	GpuErrorCheck(cudaMemcpy(d_materialsIdx     , materialsIdx , numObjects *      sizeof(int)            , cudaMemcpyHostToDevice));

	#if RENDER_SPHERE
	GpuErrorCheck(cudaMemcpy(d_spheres, spheres, numSpheres * sizeof(Float4), cudaMemcpyHostToDevice));
	#endif
	#if RENDER_SPHERE_LIGHT
	GpuErrorCheck(cudaMemcpy(d_sphereLights     , sphereLights , numSphereLights * sizeof(Float4)         , cudaMemcpyHostToDevice));
	#endif

	// setup scene
	#if RENDER_SPHERE
	d_sceneGeometry.numSpheres      = numSpheres;
	d_sceneGeometry.spheres         = d_spheres;
	#endif
	#if RENDER_SPHERE_LIGHT
	d_sceneMaterial.numSphereLights = numSphereLights;

	d_sceneMaterial.sphereLights    = d_sphereLights;
	#endif

	d_sceneGeometry.numAabbs        = numAabbs;
	d_sceneGeometry.aabbs           = d_sceneAabbs;

	d_sceneMaterial.materials       = d_surfaceMaterials;
	d_sceneMaterial.materialsIdx    = d_materialsIdx;
	d_sceneMaterial.numMaterials    = numMaterials;

	d_sceneGeometry.triangles       = triangles;
	d_sceneGeometry.bvhNodes        = bvhNodes;
	d_sceneGeometry.tlasBvhNodes    = tlasBvhNodes;
	d_sceneGeometry.numTriangles    = triCount;

	delete[] materials;
	delete[] materialsIdx;

	// cuda random
	h_randGen.init();
	d_randGen = h_randGen;

	// camera
	CameraSetup(cbo.camera);

	// textures
	texArrayUv = LoadTextureRgba8(g_settings->inputTextureFileNames[0].c_str(), sceneTextures.uv);
	//texArraySandAlbedo = LoadTextureRgb8("resources/textures/sand.png", sceneTextures.sandAlbedo);
	//texArraySandNormal = LoadTextureRgb8("resources/textures/sand_n.png", sceneTextures.sandNormal);

	// timer init
	timer.init();

	// set render dim to screen dim
	//renderWidth = screenWidth;
	//renderHeight = screenHeight;
}

void RayTracer::CameraSetup(Camera& camera)
{
	//cameraFocusPos = Float3(0, 1.0f, 0);
	//camera.pos = cameraFocusPos + Float3(7.3f, 2.0f, -6.9f);
	camera.pos = Float3(-2.f, 2.0f, -2.0f);

	//Float3 cameraLookAtPoint = cameraFocusPos;
	//Float3 camToObj = cameraLookAtPoint - camera.pos;

	//camera.dir = normalize(camToObj);
	camera.yaw = 0;
	camera.pitch = 0;
	camera.up  = { 0.0f, 1.0f, 0.0f };

	//camera.focal = camToObj.length();
	camera.focal = 5.0f;
	camera.aperture = 0.001f;

	camera.resolution = { (float)renderWidth, (float)renderHeight };
	camera.fov.x = 90.0f * Pi_over_180;

	if (g_settings->loadCameraAtInit)
	{
		LoadCameraFromFile(g_settings->inputCameraFileName);
	}

	camera.update();
}

void Buffer2DManager::init(int renderWidth, int renderHeight, int screenWidth, int screenHeight)
{
	std::array<cudaChannelFormatDesc, Buffer2DFormatCount> format;
	std::array<UInt2, Buffer2DDimCount>                    dim;

	// ------------------------- Format --------------------------
	format[FORMAT_FLOAT4] = cudaCreateChannelDesc<float4>();
	format[FORMAT_HALF2]  = cudaCreateChannelDescHalf2();
	format[FORMAT_HALF]   = cudaCreateChannelDescHalf1();
	format[FORMAT_HALF4]  = cudaCreateChannelDescHalf4();

	// ------------------------- Dimension --------------------------

	UInt2 bufferSize4  = UInt2(divRoundUp(renderWidth, 4u), divRoundUp(renderHeight, 4u));
	UInt2 bufferSize16 = UInt2(divRoundUp(bufferSize4.x, 4u), divRoundUp(bufferSize4.y, 4u));
	UInt2 bufferSize64 = UInt2(divRoundUp(bufferSize16.x, 4u), divRoundUp(bufferSize16.y, 4u));

	dim[BUFFER_2D_RENDER_DIM]     = UInt2(renderWidth, renderHeight);
	dim[BUFFER_2D_SCREEN_DIM]     = UInt2(screenWidth, screenHeight);
	dim[BUFFER_2D_RENDER_DIM_4]   = bufferSize4;
	dim[BUFFER_2D_RENDER_DIM_16]  = bufferSize16;
	dim[BUFFER_2D_RENDER_DIM_64]  = bufferSize64;
	dim[BUFFER_2D_8x8_GRID_DIM]   = UInt2(divRoundUp(renderWidth, 8u), divRoundUp(renderHeight, 8u));
	dim[BUFFER_2D_16x16_GRID_DIM] = UInt2(divRoundUp(renderWidth, 16u), divRoundUp(renderHeight, 16u));
	dim[BUFFER_2D_SKY_DIM]        = UInt2(64, 16);

	// ------------------------- Mapping --------------------------

	std::unordered_map<Buffer2DName, Buffer2DFeature> map =
	{
		{ RenderColorBuffer             , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM    } },
		{ AccumulationColorBuffer       , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM    } },
		{ HistoryColorBuffer            , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM    } },
		{ ScaledColorBuffer             , { FORMAT_HALF4  , BUFFER_2D_SCREEN_DIM    } },
		{ ScaledAccumulationColorBuffer , { FORMAT_HALF4  , BUFFER_2D_SCREEN_DIM    } },

		{ ColorBuffer4                  , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM_4  } },
		{ ColorBuffer16                 , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM_16 } },
		{ ColorBuffer64                 , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM_64 } },
		{ BloomBuffer4                  , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM_4  } },
		{ BloomBuffer16                 , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM_16 } },

		{ NormalBuffer                  , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM    } },
		{ DepthBuffer                   , { FORMAT_HALF   , BUFFER_2D_RENDER_DIM    } },
		{ HistoryDepthBuffer            , { FORMAT_HALF   , BUFFER_2D_RENDER_DIM    } },

		{ MotionVectorBuffer            , { FORMAT_HALF2  , BUFFER_2D_RENDER_DIM        } },
		{ NoiseLevelBuffer              , { FORMAT_HALF   , BUFFER_2D_8x8_GRID_DIM      } },
		{ NoiseLevelBuffer16x16         , { FORMAT_HALF   , BUFFER_2D_16x16_GRID_DIM    } },

		{ IndirectLightColorBuffer      , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM    } },
		{ IndirectLightDirectionBuffer  , { FORMAT_HALF4  , BUFFER_2D_RENDER_DIM    } },

		{ SkyBuffer                     , { FORMAT_FLOAT4 , BUFFER_2D_SKY_DIM       } },
	};

	// -------------------------- Init buffer -------------------------

	for (int i = 0; i < Buffer2DCount; ++i)
	{
		assert(i < Buffer2DCount, "Static assert: Buffer2DCount max count error!");
		Buffer2DFeature feature = map[static_cast<Buffer2DName>(i)];
		assert(static_cast<int>(feature.format) < Buffer2DFormatCount, "Static assert: Buffer2DFormatCount max count error!");
		assert(static_cast<int>(feature.dim) < Buffer2DDimCount, "Static assert: Buffer2DDimCount max count error!");
		buffers[i].init(&format[static_cast<int>(feature.format)], dim[static_cast<int>(feature.dim)]);
	}
}

void RayTracer::cleanup()
{
	// ---------------- Destroy surface objects ----------------------
	// triangle
	cudaFree(batchCountArray);

	cudaFree(constTriangles);
	cudaFree(triangles);

	// tlas
	cudaFree(tlasAabbs);
	cudaFree(tlasMorton);
	cudaFree(tlasReorderIdx);
	cudaFree(tlasBvhNodes);

	// bvh
	cudaFree(morton);
	cudaFree(reorderIdx);
	cudaFree(bvhNodes);
	cudaFree(aabbs);

	buffer2DManager.clear();

	// ---------------------- destroy texture objects --------------------------
	cudaDestroyTextureObject(sceneTextures.uv);
	cudaFreeArray(texArrayUv);

	//cudaDestroyTextureObject(sceneTextures.sandAlbedo);
	//cudaDestroyTextureObject(sceneTextures.sandNormal);
	//if (texArraySandAlbedo != nullptr) cudaFreeArray(texArraySandAlbedo);
	//if (texArraySandNormal != nullptr) cudaFreeArray(texArraySandNormal);

	// sky
	//cudaDestroyTextureObject(skyTex);
	cudaFree(skyCdf);

	// --------------------- free other gpu buffer ----------------------------
	// exposure and histogram
	cudaFree(d_exposure);
	cudaFree(d_histogram);

	// scene
	cudaFree(d_surfaceMaterials);
	cudaFree(d_sceneAabbs);
	cudaFree(d_materialsIdx);
	#if RENDER_SPHERE
	cudaFree(d_spheres);
	delete spheres;
	#endif
	#if RENDER_SPHERE_LIGHT
	cudaFree(d_sphereLights);
	delete sphereLights;
	#endif

	cudaFree(dumpFrameBuffer);

	// random
	h_randGen.clear();

	// free cpu buffer
	delete sceneAabbs;
	delete pTerrainGenerator;
}