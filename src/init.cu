
#include "kernel.cuh"
//#include "textureLoader.h"

void RayTracer::init()
{
	// init terrain
	terrain.generateHeightMap();

	// AABB
	int numAabbs;
	AABB* aabbs = terrain.generateAabbs(numAabbs);

	// sphere
	numSpheres  = 3;
	spheres       = new Sphere[numSpheres];
	cameraFocusPos = Float3(0.0f, terrain.getHeightAt(0.0f) + 0.005f, 0.0f);
	spheres[0]            = Sphere(cameraFocusPos, 0.005f);
	spheres[1] = Sphere(Float3(0.09f, terrain.getHeightAt(0.09f) + 0.005f, 0.0f) , 0.005f);
	spheres[2] = Sphere(Float3(-0.1f, terrain.getHeightAt(-0.1f) + 0.005f, 0.0f) , 0.005f);

	// surface materials
	const int numMaterials     = 6;
	SurfaceMaterial* materials = new SurfaceMaterial[numMaterials];
	materials[0].type          = LAMBERTIAN_DIFFUSE;
	materials[0].albedo        = Float3(0.8f, 0.8f, 0.8f);
	materials[1].type          = EMISSIVE;
	materials[1].albedo        = Float3(0.01f, 0.01f, 0.01f);
	materials[2].type          = EMISSIVE;
	materials[2].albedo        = Float3(0.009f, 0.002f, 0.001f);
	materials[3].type          = EMISSIVE;
	materials[3].albedo        = Float3(0.001f, 0.002f, 0.009f);
	materials[4].type          = MICROFACET_REFLECTION;
	materials[4].albedo        = Float3(1.0f, 1.0f, 1.0f);
	materials[5].type          = PERFECT_FRESNEL_REFLECTION_REFRACTION;
	materials[5].albedo        = Float3(1.0f, 1.0f, 1.0f);

	// material index for each object
	const int numObjects = numAabbs + numSpheres;
	int* materialsIdx    = new int[numObjects];
	materialsIdx[0] = 1;
	materialsIdx[1] = 2;
	materialsIdx[2] = 3;
	for (int i = numSpheres; i < numObjects; ++i)
	{
		//if (i < numSpheres + numAabbs / 2 + 10) materialsIdx[i] = 0;
		//else materialsIdx[i] = 4;
		materialsIdx[i] = 0;
	}

	// light source
	numSphereLights = 3;
	sphereLights      = new Sphere[numSphereLights];
	sphereLights[0]           = spheres[0];
	sphereLights[1] = spheres[1];
	sphereLights[2] = spheres[2];

	// init cuda
	gpuDeviceInit(0);

	// init stream
	for (uint i = 0; i < NumStreams; i++) {
        cudaStreamCreate(&streams[i]);
	}

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
	GpuErrorCheck(cudaMallocArray(&normalBufferArray, &surfaceChannelFormatDesc, renderWidth, renderHeight, cudaArraySurfaceLoadStore));
	GpuErrorCheck(cudaMallocArray(&positionBufferArray, &surfaceChannelFormatDesc, renderWidth, renderHeight, cudaArraySurfaceLoadStore));

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;

	resDesc.res.array.array = colorBufferArrayA;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferA, &resDesc));
	resDesc.res.array.array = colorBufferArrayC;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferC, &resDesc));
	resDesc.res.array.array = colorBufferArrayB;
	GpuErrorCheck(cudaCreateSurfaceObject(&colorBufferB, &resDesc));
	resDesc.res.array.array = normalBufferArray;
	GpuErrorCheck(cudaCreateSurfaceObject(&normalBuffer, &resDesc));
	resDesc.res.array.array = positionBufferArray;
	GpuErrorCheck(cudaCreateSurfaceObject(&positionBuffer, &resDesc));

	// uint buffer
	indexBuffers.init(renderBufferSize);

	GpuErrorCheck(cudaMalloc((void**)& gHitMask    , cbo.gridSize *      sizeof(ullint)));
	GpuErrorCheck(cudaMemset(gHitMask              , 0  , cbo.gridSize * sizeof(ullint)));

	// memory alloc
	GpuErrorCheck(cudaMalloc((void**)& d_spheres          , numSpheres *       sizeof(Sphere)));
	GpuErrorCheck(cudaMalloc((void**)& d_aabbs            , numAabbs *         sizeof(AABB)));
	GpuErrorCheck(cudaMalloc((void**)& d_materialsIdx     , numObjects *       sizeof(int)));
	GpuErrorCheck(cudaMalloc((void**)& d_surfaceMaterials , numMaterials *     sizeof(SurfaceMaterial)));
	GpuErrorCheck(cudaMalloc((void**)& d_sphereLights     , numSphereLights *  sizeof(Float4)));
	GpuErrorCheck(cudaMalloc((void**)& rayState           , renderBufferSize * sizeof(RayState)));

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
	{
		RandInitVec* randDirvectors;
		curandGetDirectionVectors32(&randDirvectors, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
		GpuErrorCheck(cudaMemcpy(d_randInitVec, randDirvectors, 3 * sizeof(RandInitVec), cudaMemcpyHostToDevice));
	}

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

	// denoise cbuffer
	cbDenoise.c_phi = 1.0f;
	cbDenoise.n_phi = 0.1f;
	cbDenoise.p_phi = 0.1f;
}

void RayTracer::cleanup()
{
	delete[] spheres;
	delete[] sphereLights;

	// Destroy surface objects
    cudaDestroySurfaceObject(colorBufferA);
	cudaDestroySurfaceObject(colorBufferB);
	cudaDestroySurfaceObject(colorBufferC);
	cudaDestroySurfaceObject(normalBuffer);
	cudaDestroySurfaceObject(positionBuffer);

    // Free device memory
    cudaFreeArray(colorBufferArrayA);
	cudaFreeArray(colorBufferArrayB);
	cudaFreeArray(colorBufferArrayC);
	cudaFreeArray(normalBufferArray);
	cudaFreeArray(positionBufferArray);

	indexBuffers.cleanUp();

	cudaFree(gHitMask);

	cudaFree(d_spheres);
	cudaFree(d_surfaceMaterials);
	cudaFree(d_aabbs);
	cudaFree(d_materialsIdx);
	cudaFree(d_sphereLights);

	cudaFree(d_randInitVec);

	cudaFree(rayState);
}