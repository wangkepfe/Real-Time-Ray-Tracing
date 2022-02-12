#pragma once

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <iostream>
#include <array>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <cassert>
#include <iomanip>

#include "helper_cuda.h"
#include "linearMath.h"
#include "geometry.h"
#include "timer.h"
#include "blueNoiseRandGen.h"
#include "bvhNode.cuh"
#include "globalSettings.h"
#include "cudaError.cuh"
#include "settingParams.h"

class VoxelsGenerator;

#define PLANE_OBJECT_IDX 666666
#define ENV_LIGHT_ID 9999
#define SUN_LIGHT_ID 8888
#define DEFAULT_LIGHT_ID 7777

#define USE_INTERPOLATED_FAKE_NORMAL 0

#define DEBUG_FRAME -1
#define DUMP_FRAME_NUM -1
#define DEBUG_BVH_TRAVERSE 0
#define DEBUG_RAY_AABB_INTERSECT 0
#define DEBUG_RAY_AABB_INTERSECT_DETAIL 0
#define DEBUG_BVH_BUILD 0

#define RENDER_SPHERE 0
#define RENDER_SPHERE_LIGHT 0

#define MAX_TRIANGLE_COUNT_ALLOWED 1024 * 1024
#define MIN_TRIANGLE_COUNT_ALLOWED 2

#define SKY_WIDTH 512
#define SKY_HEIGHT 256
#define SKY_SIZE SKY_WIDTH * SKY_HEIGHT
#define SKY_SCAN_BLOCK_SIZE 512
#define SKY_SCAN_BLOCK_COUNT SKY_SIZE / SKY_SCAN_BLOCK_SIZE

#define RayMax 10e10f

// ---------------------- type define ----------------------
#define RandState curandStateScrambledSobol32_t
#define RandInitVec curandDirectionVectors32_t
#define SurfObj cudaSurfaceObject_t
#define TexObj cudaTextureObject_t

// ---------------------- struct ----------------------
struct __align__(16) Camera
{
	Float3 pos;
	float  pitch;
	Float3 dir;
	float  focal;
	Float3 left;
	float  aperture;
	Float3 up;
	float  yaw;
	Float2 resolution;
	Float2 inversedResolution;
	Float2 fov;
	Float2 tanHalfFov;
	Float3 adjustedLeft;
	float  unused3;
	Float3 adjustedUp;
	float  unused4;
	Float3 adjustedFront;
	float  unused5;
	Float3 apertureLeft;
	float  unused6;
	Float3 apertureUp;
	float  unused7;

	void update()
	{
		dir = Float3(sinf(yaw) * cosf(pitch), sinf(pitch), cosf(yaw) * cosf(pitch));

		inversedResolution = 1.0f / resolution;
		fov.y = fov.x / resolution.x * resolution.y;
		tanHalfFov = Float2(tanf(fov.x / 2), tanf(fov.y / 2));

		up = Float3(0, 1, 0);
		left = normalize(cross(up, dir));
		up = normalize(cross(dir, left));

		adjustedFront = dir * focal;
		adjustedLeft = left * tanHalfFov.x * focal;
		adjustedUp = up * tanHalfFov.y * focal;

		apertureLeft = left * aperture;
		apertureUp = up * aperture;
	}

	inline __device__ __host__ Float2 WorldToScreenSpace(Float3 worldPos)
	{
		Mat3 invCamMat(left, up, dir);                                   // build view matrix
		invCamMat.transpose();                                           // orthogonal matrix, inverse is transpose
		Float3 viewSpacePos   = invCamMat * (worldPos - pos);            // transform world pos to view space
		Float2 screenPlanePos = viewSpacePos.xy / viewSpacePos.z;        // projection onto plane
		Float2 ndcSpacePos    = screenPlanePos / tanHalfFov;             // [-1, 1]
		Float2 screenSpacePos = Float2(0.5) - ndcSpacePos * Float2(0.5); // [0, 1]
		return screenSpacePos;
	}
};

struct __align__(16) HistoryCamera
{
	inline __device__ __host__ void Setup(const Camera& cam)
	{
		invCamMat = Mat3(cam.left, cam.up, cam.dir);  // build view matrix
		invCamMat.transpose();                      // orthogonal matrix, inverse is transpose
		pos = cam.pos;
	}

	inline __device__ __host__ Float2 WorldToScreenSpace(Float3 worldPos, Float2 tanHalfFov)
	{
		Float3 viewSpacePos   = invCamMat * (worldPos - pos);            // transform world pos to view space
		Float2 screenPlanePos = viewSpacePos.xy / viewSpacePos.z;        // projection onto plane
		Float2 ndcSpacePos    = screenPlanePos / tanHalfFov;             // [-1, 1]
		Float2 screenSpacePos = Float2(0.5) - ndcSpacePos * Float2(0.5); // [0, 1]
		return screenSpacePos;
	}

	Mat3 invCamMat;
	Float3 pos;
};

struct __align__(16) SceneGeometry
{
	#if RENDER_SPHERE
	Sphere*   spheres;
	int       numSpheres;
	#endif

	AABB*     aabbs;
	Triangle* triangles;
	BVHNode*  bvhNodes;
	BVHNode*  tlasBvhNodes;
	int       numAabbs;
	int       numTriangles;
};

enum SurfaceMaterialType : uint
{
	LAMBERTIAN_DIFFUSE                    = 0,
	PERFECT_REFLECTION                    = 1,
	PERFECT_FRESNEL_REFLECTION_REFRACTION = 2,
	MICROFACET_REFLECTION                 = 3,
	EMISSIVE                              = 4,
	MAT_SKY                               = 5,
};

struct __align__(16) SurfaceMaterial
{
	__device__ __host__ SurfaceMaterial() :
		albedo  {Float3(0.8f)},
		type    {PERFECT_REFLECTION},
		useTex0 {false},
		useTex1 {false},
		useTex2 {false},
		useTex3 {false},
		texId0  {0},
		texId1  {0},
		texId2  {0},
		texId3  {0},
		F0      {Float3(0.56f, 0.57f, 0.58f)},
		alpha   {0.05f}
	{}

	Float3 albedo;
	uint   type;

	bool   useTex0;
	bool   useTex1;
	bool   useTex2;
	bool   useTex3;

	uint   texId0;
	uint   texId1;
	uint   texId2;
	uint   texId3;

	Float3 F0;
	float  alpha;
};

struct __align__(16) SceneMaterial
{
	SurfaceMaterial* materials;
	int*             materialsIdx;
	int              numMaterials;

	#if RENDER_SPHERE_LIGHT
	Sphere*          sphereLights;
	int              numSphereLights;
	#endif
};

struct __align__(16) ConstBuffer
{
	Camera        camera;

	HistoryCamera historyCamera;

	Float3        sunDir;
	float         clockTime;

	int           frameNum;
	int           bvhDebugLevel;

	int           bvhBatchSize;
};

struct __align__(16) RayState
{
	Float3     orig;
	int        sampleId;

	Float3     dir;
	int        matId;

	Float3     L;
	bool       isRayIntoSurface;

	Float3     beta0;
	float      offset;

	Float3     pos;
	int        matType;

	Float3     normal;
	bool       hitLight;

	Float3     fakeNormal;
	int        lightIdx;

	Int2       idx;
	Float2     uv;

	int        i;
	bool       hit;
	float      normalDotRayDir;
	bool       isDiffuseRay;

	Float3     tangent;
	int        unusued;

	float      depth;
	bool       isDiffuse;
	int        objectIdx;
	bool       isHitProcessed;

	Float4     rand;

	bool       isOccluded;
	bool       isShadowRay;
	float      cosWi;
	float      cosWo;

	Float3     beta1;
};

union SceneTextures
{
	struct
	{
		TexObj uv;
		TexObj sandAlbedo;
		TexObj sandNormal;
	};
	TexObj array[3];
};

enum Buffer2DName
{
	RenderColorBuffer,             // main render buffer
	AccumulationColorBuffer,       // accumulation render buffer
	HistoryColorBuffer,
	ScaledColorBuffer,             // scaled to screen size
	ScaledAccumulationColorBuffer, // accumulation of scaled to screen size buffer

	ColorBuffer4,                  // 1/4 res color buffer
	ColorBuffer16,                 // 1/6 res color buffer
	ColorBuffer64,                 // 1/64 res color buffer
	BloomBuffer4,                  // 1/4 res bloom buffer
	BloomBuffer16,                 // 1/16 bloom buffer

	NormalBuffer,                  // normalBu buffer
	DepthBuffer,                   // depth buffer
	HistoryDepthBuffer,            // depth buffer

	MotionVectorBuffer,            // motion vector buffer
	NoiseLevelBuffer,              // noise level
	NoiseLevelBuffer16x16,

	SkyBuffer,                     // sky

	Buffer2DCount,
};

enum Buffer2DFormat
{
	FORMAT_FLOAT4,
	FORMAT_HALF,
	FORMAT_HALF2,
	FORMAT_HALF4,
	Buffer2DFormatCount,
};

enum Buffer2DDim
{
	BUFFER_2D_RENDER_DIM,
	BUFFER_2D_SCREEN_DIM,
	BUFFER_2D_RENDER_DIM_4,
	BUFFER_2D_RENDER_DIM_16,
	BUFFER_2D_RENDER_DIM_64,
	BUFFER_2D_8x8_GRID_DIM,
	BUFFER_2D_16x16_GRID_DIM,
	BUFFER_2D_SKY_DIM,
	Buffer2DDimCount,
};

struct Buffer2D
{
	void init(cudaChannelFormatDesc* pFormat,
	          UInt2                  dim,
	          uint                   usageFlag = cudaArraySurfaceLoadStore)
	{
		GpuErrorCheck(cudaMallocArray(&array, pFormat, dim.x, dim.y, usageFlag));

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;
		GpuErrorCheck(cudaCreateSurfaceObject(&buffer, &resDesc));
	}

	void clear()
	{
		GpuErrorCheck(cudaDestroySurfaceObject(buffer));
		GpuErrorCheck(cudaFreeArray(array));
	}

	SurfObj    buffer;
	cudaArray* array;
};

struct Buffer2DManager
{
	struct Buffer2DFeature
	{
		Buffer2DFormat format;
	    Buffer2DDim    dim;
	};

	void init(int renderWidth, int renderHeight, int screenWidth, int screenHeight);

	void clear() { for (auto& buffer : buffers) { buffer.clear(); } }

	std::array<Buffer2D, Buffer2DCount> buffers;
};

class RayTracer
{
public:

    RayTracer(
		int screenWidth,
		int screenHeight)
		:
		screenWidth {screenWidth},
		screenHeight {screenHeight}
	{}

    ~RayTracer()
	{
		cleanup();
	}

	void init(cudaStream_t* streams);
	void draw(SurfObj* d_renderTarget);
	void cleanup();

	void keyboardUpdate(int key, int scancode, int action, int mods);
	void cursorPosUpdate(double xpos, double ypos);
	void scrollUpdate(double xoffset, double yoffset);
	void mouseButtenUpdate(int button, int action, int mods);
	void SaveCameraToFile(const std::string &camFileName);
	void LoadCameraFromFile(const std::string &camFileName);

	SurfObj GetBuffer2D(Buffer2DName name) { return buffer2DManager.buffers[(uint)name].buffer; }

	const Camera& GetCamera() { return cbo.camera; }
	uint GetTriangleCount() { return triCountPadded; }

	// UI render settings
	int cursorReset = 1;
	SkyParams skyParams = {};
	RenderPassSettings renderPassSettings = {};
	PostProcessParams postProcessParams = {};

private:

	void CameraSetup(Camera& camera);
	void UpdateFrame();
	void InputControlUpdate();
	void BuildBvhLevel1();
	void BuildBvhLevel2();
	void TemporalSpatialDenoising(Int2 bufferDim, Int2 historyDim);
	void PostProcessing(Int2 bufferDim, Int2 outputDim);

	// resolution
	const int                   screenWidth;
	const int                   screenHeight;

	int                         renderWidth;
	int                         renderHeight;

	int                         historyRenderWidth;
	int                         historyRenderHeight;

	int                         maxRenderWidth;
	int                         maxRenderHeight;

	// kernel dimension
	dim3                        blockDim;
	dim3                        gridDim;

	dim3                        scaleBlockDim;
	dim3                        scaleGridDim;

	dim3                        gridDim4;
	dim3                        gridDim16;
	dim3                        gridDim64;

	UInt2                       bufferSize4;
	UInt2                       bufferSize16;
	UInt2                       bufferSize64;

	// constant buffer
	ConstBuffer                 cbo;

	// primitives
	#if RENDER_SPHERE
	Sphere*                     d_spheres;
	#endif
	#if RENDER_SPHERE_LIGHT
	Sphere*                     d_sphereLights;
	#endif
	AABB*                       d_sceneAabbs;

	// materials
	SurfaceMaterial*            d_surfaceMaterials;
	SceneMaterial               d_sceneMaterial;
	int*                        d_materialsIdx;

	// traversal structure
	SceneGeometry               d_sceneGeometry;

	// texture
	SceneTextures               sceneTextures;
	cudaArray*                  texArraySandAlbedo;
	cudaArray*                  texArrayUv;
	cudaArray*                  texArraySandNormal;

	// surface
	Buffer2DManager             buffer2DManager;

	// sky
	float*                      skyPdf;
	float*                      skyCdf;
	float*                      skyCdfScanTmp;

	// buffer
	float*                      d_exposure;
	uint*                       d_histogram;

	// rand gen
	BlueNoiseRandGeneratorHost  h_randGen;
	BlueNoiseRandGenerator      d_randGen;

	// timer
	Timer                       timer;

	// streams
	cudaStream_t*               streams;

	// cpu buffers
	Float3                      cameraFocusPos;
	AABB*                       sceneAabbs;
	#if RENDER_SPHERE
	Sphere*                     spheres;
	int                         numSpheres;
	#endif
	#if RENDER_SPHERE_LIGHT
	Sphere*                     sphereLights;
	int                         numSphereLights;
	#endif

	// cpu update
	Float2                      sunPos;
	Int2                        sunUv;
	Float3                      sunDir;
	float                       deltaTime;
	float                       clockTime;

	// sizes
	static const uint           BatchSize = 1024;
	static const uint           KernalBatchSize = 4;
	static constexpr uint       KernelSize = BatchSize / KernalBatchSize;

	// sizes
	uint                        triCount;
	uint                        triCountPadded;
	uint                        triCountPadded2;
	uint                        batchCount;
	uint                        batchCountPadded;

	// triangles
	uint*                       triCountArray;
	uint*                       batchCountArray;
	Triangle*                   constTriangles;
	Triangle*                   triangles;

	// bvh
	AABB*                       aabbs;
	uint*                       morton;
	uint*                       reorderIdx;
	BVHNode*                    bvhNodes;

	// TLAS top level acceleration structure
	AABB*                       tlasAabbs;
	uint*                       tlasMorton;
	uint*                       tlasReorderIdx;
	BVHNode*                    tlasBvhNodes;

	// debug
	uchar4*                     dumpFrameBuffer;

	// Voxels
	VoxelsGenerator*           pVoxelsGenerator;
};

