#pragma once

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "helper_cuda.h"
#include "linear_math.h"
#include "geometry.h"
#include "timer.h"
#include "terrain.hpp"
#include "blueNoiseRandGen.h"

#define MAGIC_NUMBER_PLANE 666666

// ---------------------- type define ----------------------
#define RandState curandStateScrambledSobol32_t
#define RandInitVec curandDirectionVectors32_t
#define SurfObj cudaSurfaceObject_t
#define TexObj cudaTextureObject_t
#define ullint unsigned long long int

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
	float  unused1;
	Float3 dir;
	float  focal;
	Float3 left;
	float  aperture;
	Float3 up;
	float  unused2;
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
		inversedResolution = 1.0f / resolution;
		fov.y = fov.x / resolution.x * resolution.y;
		tanHalfFov = Float2(tanf(fov.x / 2), tanf(fov.y / 2));

		left = normalize(cross(up, dir));
		up = normalize(cross(dir, left));

		adjustedFront = dir * focal;
		adjustedLeft = left * tanHalfFov.x * focal;
		adjustedUp = up * tanHalfFov.y * focal;

		apertureLeft = left * aperture;
		apertureUp = up * aperture;
	}
};

struct __align__(16) SceneGeometry
{
	Sphere* spheres;
	AABB*   aabbs;
	Triangle* triangles;
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
		albedo {Float3(0.8)},
		type {PERFECT_REFLECTION},
		useTex0 {false},
		useTex1 {false},
		useTex2 {false},
		useTex3 {false},
		texId0 {0},
		texId1 {0},
		texId2 {0},
		texId3 {0},
		F0 {Float3(0.56, 0.57, 0.58)},
		alpha {0.05}
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

	int maxSceneLoop;
	int frameNum;
	int aaSampleNum;
	int pad1;

	UInt2 gridDim;
	UInt2 bufferDim;

	uint bufferSize;
	uint gridSize;
	UInt2 pad2;
};

struct __align__(16) RayState
{
	Float3     orig;
	int        sampleId;

	Float3     dir;
	int        matId;

	Float3     probeDir;
	int        objectIdx;

	Float3     L;
	bool       isRayIntoSurface;

	Float3     beta;
	float      offset;

	Float3     pos;
	int        matType;

	Float3     normal;
	bool       hitLight;

	Float3     lightBeta;
	bool       isDiffuse;

	Int2       idx;
	int        i;
	bool       hit;

	float      surfaceBetaWeight;
	float      normalDotRayDir;
	Float2     uv;

	bool       isDiffuseRay;
	Float3     tangent;

	int        bounceLimit;
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
		int screenHeight,
		int renderWidth,
		int renderHeight)
		:
		screenWidth {screenWidth},
		screenHeight {screenHeight},
		renderWidth {renderWidth},
		renderHeight {renderHeight},
		renderBufferSize {renderWidth * renderHeight}
	{}

    ~RayTracer()
	{
		cleanup();
	}

	void init(cudaStream_t* streams);
	void draw(SurfObj* d_renderTarget);
	void cleanup();

private:

	void CameraSetup(Camera& camera);
	void UpdateFrame();

	// resolution
	const int                   screenWidth;
	const int                   screenHeight;

	const int                   renderWidth;
	const int                   renderHeight;

	// buffer size
	const int                   renderBufferSize;

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
	AABB*                       d_aabbs;
	Triangle*                   d_triangles;
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

	// terrain
	Terrain                     terrain;

	// streams
	cudaStream_t*               streams;

	// cpu buffers
	Float3                      cameraFocusPos;
	Sphere*                     spheres;
	Sphere*                     sphereLights;
	AABB*                       aabbs;
	Triangle*                   triangles;
	int                         numSpheres;
	int                         numSphereLights;
	int                         numTriangles;

	// cpu update
	Float2                      sunPos;
	Float3                      sunDir;
	float                       deltaTime;
	float                       clockTime;

	// save to file
	Float4*                     saveToFileBuffer;
};

