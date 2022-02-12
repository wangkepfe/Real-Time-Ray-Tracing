#include "kernel.cuh"
#include "debugUtil.h"
#include "updateGeometry.cuh"
#include "radixSort.cuh"
#include "buildBVH.cuh"

void RayTracer::BuildBvhLevel1()
{
    // ----------------------------------------------- Build bottom level BVH -------------------------------------------------
    // ------------------------------- Update Geometry -----------------------------------
    // out: triangles, aabbs, morton codes
    UpdateSceneGeometry <KernelSize, KernalBatchSize> <<< batchCount, KernelSize >>>
        (constTriangles, triangles, aabbs, morton, triCountArray, clockTime);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("constTriangles.csv"  , constTriangles  , triCountPadded);
        DebugPrintFile("triangles.csv"       , triangles       , triCountPadded);
        DebugPrintFile("aabbs.csv"           , aabbs           , triCountPadded);
        DebugPrintFile("morton.csv"          , morton          , triCountPadded);
    }
    #endif

    // ------------------------------- Radix Sort -----------------------------------
    // in: morton code; out: reorder idx
    RadixSort <KernelSize, KernalBatchSize> <<< batchCount, KernelSize >>>
        (morton, reorderIdx);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("morton2.csv", morton, triCountPadded);
        DebugPrintFile("reorderIdx.csv", reorderIdx, triCountPadded);
    }
    #endif

    // ------------------------------- Build LBVH -----------------------------------
    // in: aabbs, morton code, reorder idx; out: lbvh
    BuildLBVH <KernelSize, KernalBatchSize> <<< batchCount , KernelSize>>>
        (bvhNodes, aabbs, morton, reorderIdx, triCountArray);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("bvhNodes.csv", bvhNodes, triCountPadded);
    }
    #endif
}

void RayTracer::BuildBvhLevel2()
{
    // ----------------------------------------------- Build top level BVH -------------------------------------------------------------
    UpdateTLAS <KernelSize, KernalBatchSize, BatchSize> <<< 1 , KernelSize>>>
        (bvhNodes, tlasAabbs, tlasMorton, batchCountArray, triCountPadded);

    #if DEBUG_FRAME > 0
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_aabbs.csv"           , tlasAabbs           , batchCount);
        DebugPrintFile("TLAS_morton.csv"          , tlasMorton          , batchCount);
    }
    #endif

    RadixSort <KernelSize, KernalBatchSize> <<< 1, KernelSize >>>
        (tlasMorton, tlasReorderIdx);

    #if DEBUG_FRAME > 0
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_reorderIdx.csv"       , tlasReorderIdx      , batchCount);
        DebugPrintFile("TLAS_morton2.csv"          , tlasMorton          , batchCount);
    }
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    #endif

    BuildLBVH <KernelSize, KernalBatchSize> <<< 1 , KernelSize>>>
        (tlasBvhNodes, tlasAabbs, tlasMorton, tlasReorderIdx, batchCountArray);

    #if DEBUG_FRAME > 0
    if (cbo.frameNum == DEBUG_FRAME)
    {
        DebugPrintFile("TLAS_bvhNodes.csv"       , tlasBvhNodes      , batchCount);
    }
    GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());
    #endif
}