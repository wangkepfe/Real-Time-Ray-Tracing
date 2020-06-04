#pragma once

#include <device_launch_parameters.h>
#include <cuda.h>
//#include <curand.h>
//#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "helper_cuda.h"
#include "linear_math.h"
#include "geometry.h"
#include "timer.h"
#include "terrain.hpp"
#include "hash.cuh"

// ---------------------- type define ----------------------
#define RandState curandStateScrambledSobol32_t
#define RandInitVec curandDirectionVectors32_t
#define SurfObj cudaSurfaceObject_t
#define ullint unsigned long long int

// ---------------------- shader setting ----------------------
#define USE_PERFECT_FRESNEL_REFLECTION_REFRACTION 0
#define USE_MICROFACET_REFLECTION 0
#define RAND_DIM 7 // 7=2+2+1+2; 2 anti-aliasing, 1 fresnel / lights, 2 diffuse, 2 light

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
	Float4 pos;        // .w unused
	Float4 dir;        // .w unused
	Float4 left;       // left = up x dir, .w unused
	Float4 resolution; // (width, height, 1/width, 1/height)
	Float4 fov;        // (radius_x, radius_y, tan(radius_x / 2), tan(radius_y / 2))
};

struct __align__(16) SceneGeometry
{
	Sphere* spheres;
	AABB*   aabbs;
	int numAabbs;
	int numSpheres;
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
	float      offset;

	Float3     beta;
	float      distance;

	Float3     pos;
	int        matType;

	Float3     normal;
	bool       hitLight;

	Float3     lightBeta;
	bool       hasBsdfRay;

	Int2       idx;
	int        i;
	int        bounce;

	bool       terminated;
	bool       hasProbeRay;
	bool       isSunVisible;
	bool       isDiffuse;

	bool       isRayIntoSurface;
	bool       hit;
	bool       isDeltaLight;
	bool       isMoonVisible;

	float      surfaceBetaWeight;
	float      normalDotRayDir;
	Float2     uv;

	bool       isDiffuseRay;
	Float3     tangent;

	RandState  rdState[3];
};


__device__ __inline__ float rd(RandState* rdState) { return curand_uniform(rdState); }
__device__ __inline__ Float2 rd2(RandState* rdState1, RandState* rdState2) { return Float2(curand_uniform(rdState1), curand_uniform(rdState2)); }

struct IndexBuffers
{
	static const uint IndexBufferCount = 4;
	union {
		struct {
			uint* hitBufferA;
			uint* missBufferA;
			uint* hitBufferB;
			uint* missBufferB;
		};
		uint* buffers[IndexBufferCount];
	};

	union {
		struct {
			uint* hitBufferTopA;
			uint* missBufferTopA;
			uint* hitBufferTopB;
			uint* missBufferTopB;
		};
		uint* bufferTops[IndexBufferCount];
	};

	void init(uint renderBufferSize) {
		for (uint i = 0; i < IndexBufferCount; ++i) {
			GpuErrorCheck(cudaMalloc((void**)& buffers[i] , renderBufferSize *  sizeof(uint)));
			GpuErrorCheck(cudaMalloc((void**)& bufferTops[i] , sizeof(uint)));
		}
	}

	void memsetZero(uint renderBufferSize, cudaStream_t stream) {
		for (uint i = 0; i < IndexBufferCount; ++i) {
			GpuErrorCheck(cudaMemsetAsync(buffers[i], 0u, renderBufferSize * sizeof(uint), stream));
			//GpuErrorCheck(cudaMemsetAsync(bufferTops[i], 0u, sizeof(uint), stream));
		}
	}

	void cleanUp() {
		for (uint i = 0; i < IndexBufferCount; ++i) {
			cudaFree(buffers[i]);
			cudaFree(bufferTops[i]);
		}
	}
};

union SceneTextures
{
	struct
	{
		cudaTextureObject_t uv;
		cudaTextureObject_t sandAlbedo;
		cudaTextureObject_t sandNormal;
	};
	cudaTextureObject_t array[3];
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

	void LoadTextures();

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

	// ray state, intersection
	RayState*                   rayState;

	// primitives
	Sphere*                     d_spheres;
	AABB*                       d_aabbs;
	Sphere*                     d_sphereLights;

	// materials
	SurfaceMaterial*            d_surfaceMaterials;
	SceneMaterial               d_sceneMaterial;
	int*                        d_materialsIdx;

	// traversal structure
	SceneGeometry               d_sceneGeometry;

	//
	SceneTextures               sceneTextures;

	// buffer
	cudaArray*                  colorBufferArrayA;
	cudaArray*                  colorBufferArrayB;
	cudaArray*                  colorBufferArrayC;

	cudaArray*                  colorBufferArray4;
	cudaArray*                  colorBufferArray16;
	cudaArray*                  colorBufferArray64;

	cudaArray*                  bloomBufferArray4;
	cudaArray*                  bloomBufferArray16;

	cudaArray*                  texArraySandAlbedo;
	cudaArray*                  texArrayUv;
	cudaArray*                  texArraySandNormal;

	SurfObj         colorBufferA;
	SurfObj         colorBufferB;
	SurfObj         colorBufferC;

	SurfObj         colorBuffer4;
	SurfObj         colorBuffer16;
	SurfObj         colorBuffer64;

	SurfObj         bloomBuffer4;
	SurfObj         bloomBuffer16;

	IndexBuffers indexBuffers;
	ullint* gHitMask;

	//
	float* d_exposure;
	uint* d_histogram;

	// rand init
	RandInitVec*                d_randInitVec;

	// timer
	Timer                       timer;

	// terrain
	Terrain                     terrain;

	static const uint NumStreams = 4;
	cudaStream_t* streams;
	Float3  cameraFocusPos;
	Sphere* spheres;
	Sphere* sphereLights;
	int numSpheres;
	int numSphereLights;
};

