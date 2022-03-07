
#include "kernel.cuh"
#include "fileUtils.cuh"
#include "blueNoiseRandGenData.h"
#include "cuda_fp16.h"
#include "terrain.h"
#include "meshing.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
		// load triangles from mesh file
		// std::vector<Triangle> h_triangles
		// LoadTrianglesFromFile(h_triangles, triCount);

		// Generate terrain voxels
		pVoxelsGenerator = new VoxelsGenerator();
		pVoxelsGenerator->Generate();

		// Convert voxels to triangle mesh
		//BlockMeshGenerator blockMeshGenerator(*pVoxelsGenerator);
		//std::vector<Triangle> h_triangles = *blockMeshGenerator.VoxelToMesh();

		MarchingCubeMeshGenerator marchingCubeMeshGenerator(*pVoxelsGenerator);
		// std::vector<Triangle> h_triangles = *marchingCubeMeshGenerator.VoxelToMesh();
		std::vector<Float3> h_vertexBuffer;
		std::vector<uint> h_indexBuffer;
		marchingCubeMeshGenerator.VoxelToMesh(h_vertexBuffer, h_indexBuffer);

		// triCount = h_triangles.size();
		triCount = h_indexBuffer.size() / 3;

		// Max and min tri count supported
		assert(triCount >= MIN_TRIANGLE_COUNT_ALLOWED);
		assert(triCount <= MAX_TRIANGLE_COUNT_ALLOWED);

		// pad with repeat triangles, required by update geometry
		triCountPadded = triCount;
		if (triCountPadded % KernalBatchSize != 0)
		{
			triCountPadded += (KernalBatchSize - triCount % KernalBatchSize);
			// h_triangles.resize(triCountPadded);
			h_indexBuffer.resize(triCountPadded * 3);
			for (int i = triCount * 3; i < triCountPadded * 3; ++i)
			{
				// h_triangles[i] = h_triangles[0];
				h_indexBuffer[i] = 0;
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
		// GpuErrorCheck(cudaMalloc((void**)& constTriangles, triCountPadded * sizeof(Triangle)));
		// GpuErrorCheck(cudaMemcpy(constTriangles, h_triangles.data(), triCountPadded * sizeof(Triangle), cudaMemcpyHostToDevice));

		numVertices = h_vertexBuffer.size();
		numIndices = triCountPadded * 3;

		GpuErrorCheck(cudaMalloc((void**)& indexBuffer, numIndices * sizeof(uint)));
		GpuErrorCheck(cudaMemcpy(indexBuffer, h_indexBuffer.data(), numIndices * sizeof(uint), cudaMemcpyHostToDevice));

		GpuErrorCheck(cudaMalloc((void**)& vertexBuffer, numVertices * sizeof(Float3)));
		GpuErrorCheck(cudaMalloc((void**)& normalBuffer, numVertices * sizeof(Float3)));

		GpuErrorCheck(cudaMalloc((void**)& constVertexBuffer, numVertices * sizeof(Float3)));
		GpuErrorCheck(cudaMemcpy(constVertexBuffer, h_vertexBuffer.data(), numVertices * sizeof(Float3), cudaMemcpyHostToDevice));

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

	cbo.bvhBatchSize = BatchSize;
	cbo.bvhNodesSize = triCountPadded;
	cbo.trianglesSize = triCountPadded;
	cbo.tlasBvhNodesSize = batchCount;

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
	// materials[i].useTex0       = true;
	// materials[i].texId0        = static_cast<uint>(GroundSoilAlbedoRoughness);
	// materials[i].useTex1       = true;
	// materials[i].texId1        = static_cast<uint>(GroundSoilNormalHeight);
	// materials[i].useTex2       = true;
	// materials[i].texId0        = static_cast<uint>(GroundSoilAo);
	++i;
	materials[i].type          = MICROFACET_REFLECTION;
	materials[i].albedo        = Float3(0.9f);
	materials[i].F0            = Float3(0.56f, 0.57f, 0.58f);
	materials[i].alpha         = 0.05f;
	++i;
	materials[i].type          = PERFECT_REFLECTION;
	++i;
	materials[i].type          = LAMBERTIAN_DIFFUSE;
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

	GpuErrorCheck(cudaMalloc((void**)&skyCdf, SKY_SIZE * sizeof(float)));
	GpuErrorCheck(cudaMemset(skyCdf, 0, SKY_SIZE * sizeof(float)));

	GpuErrorCheck(cudaMalloc((void**)&skyPdf, SKY_SIZE * sizeof(float)));
	GpuErrorCheck(cudaMemset(skyPdf, 0, SKY_SIZE * sizeof(float)));

	GpuErrorCheck(cudaMalloc((void**)&skyCdfScanTmp, SKY_SCAN_BLOCK_COUNT * sizeof(float)));
	GpuErrorCheck(cudaMemset(skyCdfScanTmp, 0, SKY_SCAN_BLOCK_COUNT * sizeof(float)));

	GpuErrorCheck(cudaMalloc((void**)&sunCdf, SUN_SIZE * sizeof(float)));
	GpuErrorCheck(cudaMemset(sunCdf, 0, SUN_SIZE * sizeof(float)));

	GpuErrorCheck(cudaMalloc((void**)&sunPdf, SUN_SIZE * sizeof(float)));
	GpuErrorCheck(cudaMemset(sunPdf, 0, SUN_SIZE * sizeof(float)));

	GpuErrorCheck(cudaMalloc((void**)&sunCdfScanTmp, SUN_SCAN_BLOCK_COUNT * sizeof(float)));
	GpuErrorCheck(cudaMemset(sunCdfScanTmp, 0, SUN_SCAN_BLOCK_COUNT * sizeof(float)));

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
	d_sceneMaterial.numMaterialsIdx = numObjects;

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
	// texArrayUv = LoadTextureRgba8(g_settings->inputTextureFileNames[0].c_str(), sceneTextures.uv);
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
	std::array<UInt2, Buffer2DDimCount> dim;

	// ------------------------- Format --------------------------
	format[FORMAT_FLOAT4]  = cudaCreateChannelDesc<float4>();
	format[FORMAT_HALF2]   = cudaCreateChannelDescHalf2();
	format[FORMAT_HALF]    = cudaCreateChannelDescHalf1();
	format[FORMAT_HALF4]   = cudaCreateChannelDescHalf4();
	format[FORMAT_USHORT]  = cudaCreateChannelDesc<ushort1>();
	format[FORMAT_USHORT4] = cudaCreateChannelDesc<ushort4>();

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
	dim[BUFFER_2D_SKY_DIM]        = UInt2(SKY_WIDTH, SKY_HEIGHT);
	dim[BUFFER_2D_SUN_DIM]        = UInt2(SUN_WIDTH, SUN_HEIGHT);
	dim[BUFFER_2D_1024x1024]      = UInt2(1024, 1024);

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

		{ SkyBuffer                     , { FORMAT_FLOAT4 , BUFFER_2D_SKY_DIM       } },
		{ SunBuffer                     , { FORMAT_FLOAT4 , BUFFER_2D_SKY_DIM       } },

		{ AlbedoBuffer                  , { FORMAT_HALF4   , BUFFER_2D_RENDER_DIM    } },

		{ SoilAlbedoAoBuffer            , { FORMAT_USHORT4 , BUFFER_2D_1024x1024    } } ,
		{ SoilNormalRoughnessBuffer     , { FORMAT_USHORT4 , BUFFER_2D_1024x1024    } } ,
		{ SoilHeightBuffer              , { FORMAT_USHORT  , BUFFER_2D_1024x1024    } } ,
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

	struct TextureDescripton
	{
		const char* filepath;
		int numChannel;
	};

	std::unordered_map<Buffer2DName, TextureDescripton> textureFileMap =
	{
		{ SoilAlbedoAoBuffer            , { "resources/textures/soil/out/albedoAo.png"        , STBI_rgb_alpha }    } ,
		{ SoilNormalRoughnessBuffer     , { "resources/textures/soil/out/normalRoughness.png" , STBI_rgb_alpha }    } ,
		{ SoilHeightBuffer              , { "resources/textures/soil/out/height.png"          , STBI_grey      }    } ,
	};

	for (const auto& pair : textureFileMap)
	{
		int texWidth;
		int texHeight;
		int texChannel;

		auto buffer = stbi_load_16(pair.second.filepath, &texWidth, &texHeight, &texChannel, pair.second.numChannel);
		assert(buffer != NULL);

		GpuErrorCheck(cudaMemcpyToArray(buffers[static_cast<int>(pair.first)].array, 0, 0, buffer, texWidth * texHeight * pair.second.numChannel * 2, cudaMemcpyHostToDevice));

		STBI_FREE(buffer);
	}
}

// template<int numChannel, typename TexelType, typename LoadFunc>
// inline uint16_t* CreateTexture(const char* filepath, cudaArray*& buffer, TexObj& texObj, cudaChannelFormatDesc& channelDesc, cudaResourceDesc& resDesc, cudaTextureDesc& desc)
// {
// 	int texWidth;
// 	int texHeight;
// 	int texChannel;

// 	uint16_t* hBufferPtr = LoadFunc()(filepath, texWidth, texHeight, texChannel, numChannel);

// 	int texelSize = sizeof(TexelType);
// 	channelDesc = cudaCreateChannelDesc<TexelType>();
// 	GpuErrorCheck(cudaMallocArray(&buffer, &channelDesc, texWidth, texHeight, cudaArrayTextureGather));
// 	GpuErrorCheck(cudaMemcpyToArray(buffer, 0, 0, hBufferPtr, texWidth * texHeight * texelSize, cudaMemcpyHostToDevice));

// 	resDesc = cudaResourceDesc{};
// 	resDesc.resType = cudaResourceTypeArray;
// 	resDesc.res.array.array = buffer;

// 	desc = cudaTextureDesc{};
// 	desc.addressMode[0]   = cudaAddressModeWrap;
// 	desc.addressMode[1]   = cudaAddressModeWrap;
// 	desc.filterMode       = cudaFilterModeLinear;
// 	desc.readMode         = cudaReadModeNormalizedFloat;
// 	desc.normalizedCoords = 1;

// 	GpuErrorCheck(cudaCreateTextureObject(&texObj, &resDesc, &desc, NULL));
// 	assert(texObj != NULL);

// 	return hBufferPtr;
// }

// void TextureManager::init()
// {
// 	std::unordered_map<TexName, TextureDesc> map =
// 	{
// 		{ GroundSoilAlbedoRoughness, { "resources/textures/soil/out/albedoRoughness.png", RGBA16 } },
// 		{ GroundSoilNormalHeight   , { "resources/textures/soil/out/normalHeight.png"   , RGBA16 } },
// 		{ GroundSoilAo             , { "resources/textures/soil/out/ao.png"             , RGBA16  } },
// 	};

// 	for (int i = 0; i < TexName::TextureCount; ++i)
// 	{
// 		auto name = static_cast<TexName>(i);
// 		auto& texDesc = map[name];

// 		hBuffers[i] = CreateTexture<4, ushort4, LoadTexture16>(texDesc.filepath.c_str(), buffers[i], texObj.array[i], channelDescs[i], resDescs[i], texDescs[i]);

// 		// switch (texDesc.type)
// 		// {
// 		// 	case RGBA16: CreateTexture<4, ushort4, LoadTexture16>(texDesc.filepath.c_str(), buffers[i], texObj.array[i], channelDescs[i], resDescs[i], texDescs[i]); break;
// 		// 	case RGBA8:  CreateTexture<4, uchar4,  LoadTexture8> (texDesc.filepath.c_str(), buffers[i], texObj.array[i], channelDescs[i], resDescs[i], texDescs[i]); break;
// 		// }
// 	}
// }

// void TextureManager::init()
// {
// 	std::unordered_map<TexName, TextureDesc> map =
// 	{
// 		{ GroundSoilAlbedoRoughness, { "resources/textures/soil/out/albedoRoughness.png", RGBA16 } },
// 		{ GroundSoilNormalHeight   , { "resources/textures/soil/out/normalHeight.png"   , RGBA16 } },
// 		{ GroundSoilAo             , { "resources/textures/soil/out/ao.png"             , RGBA16 } },
// 	};

// 	for (int i = 0; i < TexName::TextureCount; ++i)
// 	{
// 		auto name = static_cast<TexName>(i);
// 		auto& texDesc = map[name];

// 		int texWidth;
// 		int texHeight;
// 		int texChannel;

// 		hBuffers[i] = LoadTexture16()(texDesc.filepath.c_str(), texWidth, texHeight, texChannel, 4);

// 		channelDescs[i] = cudaCreateChannelDesc<ushort4>();
// 		int texelSize = sizeof(ushort4);

// 		GpuErrorCheck(cudaMallocArray(&buffers[i], &channelDescs[i], texWidth, texHeight));
// 		GpuErrorCheck(cudaMemcpyToArray(buffers[i], 0, 0, hBuffers[i], texWidth * texHeight * texelSize, cudaMemcpyHostToDevice));

// 		memset(&resDescs[i], 0, sizeof(cudaResourceDesc));
// 		resDescs[i].resType            = cudaResourceTypeArray;
// 		resDescs[i].res.array.array    = buffers[i];

// 		memset(&texDescs[i], 0, sizeof(cudaTextureDesc));
// 		texDescs[i].normalizedCoords = 1;
//         texDescs[i].filterMode = cudaFilterModeLinear;
//         texDescs[i].addressMode[0] = cudaAddressModeWrap;
//         texDescs[i].addressMode[1] = cudaAddressModeWrap;
//         texDescs[i].addressMode[2] = cudaAddressModeWrap;
// 		texDescs[i].readMode = cudaReadModeNormalizedFloat;

// 		GpuErrorCheck(cudaCreateTextureObject(&texObj.array[i], &resDescs[i], &texDescs[i], NULL));
// 	}
// }

void RayTracer::cleanup()
{
	// ---------------- Destroy surface objects ----------------------
	// triangle
	cudaFree(batchCountArray);

	cudaFree(vertexBuffer);
	cudaFree(normalBuffer);
	cudaFree(indexBuffer);
	cudaFree(constVertexBuffer);
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
	// textureManager.clear();

	// ---------------------- destroy texture objects --------------------------

	//cudaDestroyTextureObject(sceneTextures.sandAlbedo);
	//cudaDestroyTextureObject(sceneTextures.sandNormal);
	//if (texArraySandAlbedo != nullptr) cudaFreeArray(texArraySandAlbedo);
	//if (texArraySandNormal != nullptr) cudaFreeArray(texArraySandNormal);

	// sky
	//cudaDestroyTextureObject(skyTex);
	cudaFree(skyCdf);
	cudaFree(skyPdf);
	cudaFree(skyCdfScanTmp);
	cudaFree(sunCdf);
	cudaFree(sunPdf);
	cudaFree(sunCdfScanTmp);

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
	delete pVoxelsGenerator;
}