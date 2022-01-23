#pragma once

#include <vector>
#include <memory>

#include "linear_math.h"
#include "geometry.h"

class VoxelsGenerator;

//--------------------------------------------------------------------------------------------------------------------------------------------
// Voxel Mesh Generator
class VoxelMeshGenerator
{
public:
    VoxelMeshGenerator(const VoxelsGenerator& voxels) : voxels {voxels} {}
    virtual ~VoxelMeshGenerator() {}

    virtual std::shared_ptr<std::vector<Triangle>> VoxelToMesh() = 0;
    
protected:
    const VoxelsGenerator& voxels;
};

//--------------------------------------------------------------------------------------------------------------------------------------------
// Block Mesh Generator
class BlockMeshGenerator : public VoxelMeshGenerator
{
public:
    BlockMeshGenerator(const VoxelsGenerator& voxels) : VoxelMeshGenerator(voxels) {}
    virtual ~BlockMeshGenerator() {}    

    virtual std::shared_ptr<std::vector<Triangle>> VoxelToMesh() override;
};

//--------------------------------------------------------------------------------------------------------------------------------------------
// Marching Cube Mesh Generator
class MarchingCubeMeshGenerator : public VoxelMeshGenerator
{
public:
    MarchingCubeMeshGenerator(const MarchingCubeMeshGenerator& voxels) : VoxelMeshGenerator(voxels) {}
    virtual ~MarchingCubeMeshGenerator() {}    

    virtual std::shared_ptr<std::vector<Triangle>> VoxelToMesh() override
    {
        return nullptr;
    }
};