#pragma once

#include <vector>
#include <memory>
#include <string>

#include "linear_math.h"
#include "geometry.h"

class VoxelsGenerator;

enum class Axis
{
    pX = 0x1,
    pY = 0x2,
    pZ = 0x4,
    nX = 0x8,
    nY = 0x10,
    nZ = 0x20,
};

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
private:
    struct Transformation
    {
        enum Type
        {
            None,
            Mirror,
            Rotate,
        } type;
        Axis axis;
        int angle = 0;
        int basedOnId = 0;
    };

    struct MarchingCube
    {
        std::string filename;
        std::vector<Float3> points;
        bool reversible = false;
    };
public:
    MarchingCubeMeshGenerator(const VoxelsGenerator& voxels) : VoxelMeshGenerator(voxels) { Init(); }
    virtual ~MarchingCubeMeshGenerator() {}

    virtual std::shared_ptr<std::vector<Triangle>> VoxelToMesh() override;
private:
    void Init();
    void InitMarchingCube(const MarchingCube& marchingCube);

    std::vector<Transformation> transList;
    std::vector<std::vector<Triangle>> meshes;
};