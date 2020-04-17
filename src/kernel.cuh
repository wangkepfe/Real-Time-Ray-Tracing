#pragma once

#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "helper_cuda.h"
#include "linear_math.h"
#include "geometry.h"
#include "timer.h"
#include "terrain.hpp"

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
	Sphere* sphereLights;
	uint pad16;

	int numAabbs;
	int numSpheres;
	int numSphereLights;
	uint pad32;
};

enum SurfaceMaterialType
{
	LAMBERTIAN_DIFFUSE                    = 0,
	PERFECT_REFLECTION                    = 1,
	PERFECT_FRESNEL_REFLECTION_REFRACTION = 2,
	MICROFACET_REFLECTION                 = 3,
	EMISSIVE                              = 4,
};

struct __align__(16) SurfaceMaterial
{
	Float3 albedo;
	float  type;
};

struct __align__(16) SceneMaterial
{
	SurfaceMaterial* materials;
	int* materialsIdx;

	int numMaterials;
	uint pad16;
};

struct __align__(16) ConstBuffer
{
	Camera camera;

	Float3 sunDir;
	float  pad0;

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
	Ray        ray;                 // ray orig, dir

	Float3     L;                   // current light contribution
	float      offset;              // ray offset
	Float3     beta;                // color mask
	float      pad0;
	Float3     pos;                 // ray intersect point
	float      pad1;
	Float3     normal;              // ray intersect normal
	float      pad2;

	Int2       idx;                 // pixel idx
	int        bounce;              // how many bounces
	int        objectIdx;           // intersection object idx
	bool       hit;                 // hit or miss
	bool       terminated;
	Int2       pad3;

	RandState  rdState[3];          // quasi rand state
};


__device__ __inline__ float rd(RandState* rdState) { return curand_uniform(rdState); }
__device__ __inline__ Float2 rd2(RandState* rdState1, RandState* rdState2) { return Float2(curand_uniform(rdState1), curand_uniform(rdState2)); }

struct __align__(16) CbDenoise
{
	float c_phi, n_phi, p_phi, pad16;
};

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

	void memsetZero(uint renderBufferSize) {
		for (uint i = 0; i < IndexBufferCount; ++i) {
			GpuErrorCheck(cudaMemsetAsync(buffers[i], 0u, renderBufferSize * sizeof(uint)));
			GpuErrorCheck(cudaMemsetAsync(bufferTops[i], 0u, sizeof(uint)));
		}
	}

	void cleanUp() {
		for (uint i = 0; i < IndexBufferCount; ++i) {
			cudaFree(buffers[i]);
			cudaFree(bufferTops[i]);
		}
	}
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

	void init();
	void draw(SurfObj* d_renderTarget);
	void cleanup();

private:

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

	// constant buffer
	ConstBuffer                 cbo;
	CbDenoise                   cbDenoise;

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

	// buffer
	cudaArray*                  colorBufferArrayA;
	cudaArray*                  colorBufferArrayB;
	cudaArray*                  colorBufferArrayC;
	cudaArray*                  normalBufferArray;
	cudaArray*                  positionBufferArray;

	SurfObj         colorBufferA;
	SurfObj         colorBufferB;
	SurfObj         colorBufferC;
	SurfObj         normalBuffer;
	SurfObj         positionBuffer;

	IndexBuffers indexBuffers;
	ullint* gHitMask;

	// rand init
	RandInitVec*                d_randInitVec;

	// timer
	Timer                       timer;

	// terrain
	Terrain                     terrain;
};

