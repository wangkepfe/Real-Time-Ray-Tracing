#pragma once

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <iostream>

#include "helper_cuda.h"
#include "linear_math.h"
#include "geometry.h"
#include "timer.h"
#include "blueNoiseRandGen.h"
#include "bvhNode.cuh"

#define PLANE_OBJECT_IDX 666666
#define ENV_LIGHT_ID 9999
#define SUN_LIGHT_ID 8888
#define DEFAULT_LIGHT_ID 7777

#define USE_INTERPOLATED_FAKE_NORMAL 0

const bool UseDynamicResolution = 1;

// ---------------------- type define ----------------------
#define RandState curandStateScrambledSobol32_t
#define RandInitVec curandDirectionVectors32_t
#define SurfObj cudaSurfaceObject_t
#define TexObj cudaTextureObject_t

// ---------------------- error handling ----------------------
#define CheckCurandErrors(x) do { if((x)!=CURAND_STATUS_SUCCESS) { printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)
#define GpuErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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
	Sphere* spheres;
	AABB*   aabbs;
	Triangle* triangles;
	BVHNode* bvhNodes;
	int numAabbs;
	int numSpheres;
	int numTriangles;
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
		albedo {Float3(0.8f)},
		type {PERFECT_REFLECTION},
		useTex0 {false},
		useTex1 {false},
		useTex2 {false},
		useTex3 {false},
		texId0 {0},
		texId1 {0},
		texId2 {0},
		texId3 {0},
		F0 {Float3(0.56f, 0.57f, 0.58f)},
		alpha {0.05f}
	{}

	Float3 albedo;
	uint  type;

	bool useTex0;
	bool useTex1;
	bool useTex2;
	bool useTex3;

	uint texId0;
	uint texId1;
	uint texId2;
	uint texId3;

	Float3 F0;
	float alpha;
};

struct __align__(16) SceneMaterial
{
	SurfaceMaterial* materials;
	int* materialsIdx;
	Sphere* sphereLights;
	int numMaterials;
	int numSphereLights;
};

struct __align__(16) ConstBuffer
{
	Camera camera;

	Float3 sunDir;
	float  clockTime;

	int frameNum;
	int bvhDebugLevel;

	HistoryCamera historyCamera;
};

struct __align__(16) RayState
{
	Float3     orig;
	int        sampleId;

	Float3     dir;
	int        matId;

	Float3     L;
	bool       isRayIntoSurface;

	Float3     beta;
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
	int        bounceLimit;

	float      depth;
	bool       isDiffuse;
	int        objectIdx;
};

//__device__ __inline__ float rd(RandState* rdState) { return curand_uniform(rdState); }
//__device__ __inline__ Float2 rd2(RandState* rdState1, RandState* rdState2) { return Float2(curand_uniform(rdState1), curand_uniform(rdState2)); }

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

private:

	void CameraSetup(Camera& camera);
	void UpdateFrame();
	void InputControlUpdate();

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
	Sphere*                     d_spheres;
	AABB*                       d_sceneAabbs;
	Sphere*                     d_sphereLights;

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
	SurfObj                     colorBufferA;
	SurfObj                     colorBufferB;
	cudaArray*                  colorBufferArrayA;
	cudaArray*                  colorBufferArrayB;

	SurfObj                     colorBuffer4;
	SurfObj                     colorBuffer16;
	SurfObj                     colorBuffer64;
	cudaArray*                  colorBufferArray4;
	cudaArray*                  colorBufferArray16;
	cudaArray*                  colorBufferArray64;

	SurfObj                     bloomBuffer4;
	SurfObj                     bloomBuffer16;
	cudaArray*                  bloomBufferArray4;
	cudaArray*                  bloomBufferArray16;

	SurfObj                     normalDepthBufferA;
	SurfObj                     normalDepthBufferB;
	cudaArray*                  normalDepthBufferArrayA;
	cudaArray*                  normalDepthBufferArrayB;

	SurfObj                     motionVectorBuffer;
	cudaArray*                  motionVectorBufferArray;

	SurfObj                     sampleCountBuffer;
	cudaArray*                  sampleCountBufferArray;

	SurfObj                     noiseLevelBuffer;
	cudaArray*                  noiseLevelBufferArray;

	// sky
	const unsigned int          skyWidth = 64;
	const unsigned int          skyHeight = 16;
	const unsigned int          skySize = 1024;

	SurfObj                     skyBuffer;
	TexObj                      skyTex;
	cudaArray*                  skyArray;
	float*                      skyCdf;

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
	Sphere*                     spheres;
	Sphere*                     sphereLights;
	AABB*                       sceneAabbs;
	int                         numSpheres;
	int                         numSphereLights;

	// cpu update
	Float2                      sunPos;
	Int2                        sunUv;
	Float3                      sunDir;
	float                       deltaTime;
	float                       clockTime;

	// bvh and triangles
	uint       triCount;
	static const uint BVHcapacity = 1024;
	Triangle*  constTriangles;
	Triangle*  triangles;
	AABB*      aabbs;
	AABB*      sceneBoundingBox;
	uint*      morton;
	uint*      reorderIdx;
	BVHNode*   bvhNodes;
	uint*      bvhNodeParents;
	uint*      isAabbDone;

	//
	uchar4* dumpFrameBuffer;
};

