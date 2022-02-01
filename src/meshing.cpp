#include "meshing.h"
#include "terrain.h"
#include "mesh.h"
#include "fileUtils.cuh"
#include <unordered_map>
#include <iostream>

namespace
{

enum Axis
{
    pX = 0x1, 
    pY = 0x2, 
    pZ = 0x4, 
    nX = 0x8, 
    nY = 0x10, 
    nZ = 0x20,
};

struct QuadFace
{
    Float3 point;
    Axis axis;
};

struct QuadFaceHasher 
{
    unsigned long long operator() (const QuadFace& v) const 
    {
        return Float3Hasher()(v.point);
    }
};

struct QuadFaceEqualOperator
{
    bool operator() (const QuadFace& a, const QuadFace& b) const
    {
        return a.point == b.point;
    }
};

inline bool isEmpty(uint block)
{
    return block == 0 || block == 0xFFFF;
}

inline bool isSolid(uint block)
{
    return !isEmpty(block);
}

enum class MarchingCubeType
{
    t1_px_py_pz,
    t1_nx_py_pz,
    t1_px_ny_pz,
    t1_nx_ny_pz,
    t1_px_py_nz,
    t1_nx_py_nz,
    t1_px_ny_nz,
    t1_nx_ny_nz,
    count,
};

inline Float3 PointRotate(const Float3& v, Axis axis, int cosTheta, int sinTheta)
{
    if (axis == Axis::pY)
    {
        return Float3(cosTheta * v.x + sinTheta * v.z , v.y, - sinTheta * v.x + cosTheta * v.z);
    }
    else if (axis == Axis::pX)
    {
        return Float3(v.x, cosTheta * v.y - sinTheta * v.z, sinTheta * v.y + cosTheta * v.z);
    }
    else if (axis == Axis::pZ)
    {
        return Float3(cosTheta * v.x - sinTheta * v.y , sinTheta * v.x + cosTheta * v.y, v.z);
    }
}

inline void MeshRotate(std::vector<Triangle>& out, const std::vector<Triangle>& in, Axis axis, int cosTheta, int sinTheta)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
    {
        out[i].v1 = PointRotate(in[i].v1, axis, cosTheta, sinTheta);
        out[i].v2 = PointRotate(in[i].v2, axis, cosTheta, sinTheta);
        out[i].v3 = PointRotate(in[i].v3, axis, cosTheta, sinTheta);  
    }
}

inline Float3 PointMirror(const Float3& v, Axis axis)
{
    if (axis == Axis::pY)
    {
        return Float3(v.x, -v.y, v.z);
    }
    else if (axis == Axis::pX)
    {
        return Float3(-v.x, v.y, v.z);
    }
    else if (axis == Axis::pZ)
    {
        return Float3(v.x, v.y, -v.z);
    }
}

inline void MeshMirror(std::vector<Triangle>& out, const std::vector<Triangle>& in, Axis axis)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
    {
        out[i].v1 = PointMirror(in[i].v1, axis);
        out[i].v2 = PointMirror(in[i].v2, axis);
        out[i].v3 = PointMirror(in[i].v3, axis);
    }
}

inline void MeshTraslate(std::vector<Triangle>& out, const std::vector<Triangle>& in, const Float3& t)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
    {
        out[i].v1 = in[i].v1 + t;
        out[i].v2 = in[i].v2 + t;
        out[i].v3 = in[i].v3 + t;
    }
}

inline void MeshScale(std::vector<Triangle>& out, const std::vector<Triangle>& in, float scale)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
    {
        out[i].v1 = in[i].v1 * scale;
        out[i].v2 = in[i].v2 * scale;
        out[i].v3 = in[i].v3 * scale;
    }
}

} // namespace

std::shared_ptr<std::vector<Triangle>> BlockMeshGenerator::VoxelToMesh()
{
    std::unordered_set<QuadFace, QuadFaceHasher, QuadFaceEqualOperator> faces;

    for (uint i = 0; i < voxels.kMapDim; ++i)
    {
        for (uint j = 0; j < voxels.kMapDim; ++j)
        {
            for (uint k = 0; k < voxels.kMapDimY; ++k)
            {
                uint block = voxels.GetBlockAt(i, k, j);
                
                if (block == 1)
                {
					std::vector<uint> neighbors = voxels.GetNeighborBlockAt(i, k, j);

					float x = static_cast<float>(i);
					float y = static_cast<float>(k);
					float z = static_cast<float>(j);

                    QuadFace qf[6] = 
                    {
                        { Float3 (x + 0.5f, y + 0.0f, z + 0.5f), Axis::nY },
                        { Float3 (x + 0.5f, y + 1.0f, z + 0.5f), Axis::pY },
                        { Float3 (x + 0.0f, y + 0.5f, z + 0.5f), Axis::nX },
                        { Float3 (x + 1.0f, y + 0.5f, z + 0.5f), Axis::pX },
                        { Float3 (x + 0.5f, y + 0.5f, z + 0.0f), Axis::nZ },
                        { Float3 (x + 0.5f, y + 0.5f, z + 1.0f), Axis::pZ },
                    };

                    for (int ii = 0; ii < 6; ++ii)
                    {
                        if (neighbors[ii] == 0 || neighbors[ii] == 0xFFFF)
                        {
                            faces.insert(qf[ii]);
                        }
                    }
                }
            }
        }
    }

    std::vector<std::vector<size_t>> polygons;
    std::vector<Float3> verts;

    std::unordered_map<Float3, size_t, Float3Hasher> vertsMap;

    size_t idx = 0;

    for (const auto& face : faces)
    {
        Float3 vert[4];

        if (face.axis == Axis::pY)
        {
            vert[0] = face.point + Float3(0.5f, 0, 0.5f);
            vert[1] = face.point + Float3(0.5f, 0, -0.5f);
            vert[2] = face.point + Float3(-0.5f, 0, -0.5f);
            vert[3] = face.point + Float3(-0.5f, 0, 0.5f);
        }
        else if (face.axis == Axis::nY)
        {
            vert[0] = face.point + Float3(0.5f, 0, 0.5f);
            vert[1] = face.point + Float3(-0.5f, 0, 0.5f);
            vert[2] = face.point + Float3(-0.5f, 0, -0.5f);
            vert[3] = face.point + Float3(0.5f, 0, -0.5f);
        }
        else if (face.axis == Axis::pX)
        {
            vert[0] = face.point + Float3(0, 0.5f, 0.5f);
            vert[1] = face.point + Float3(0, -0.5f, 0.5f);
            vert[2] = face.point + Float3(0, -0.5f, -0.5f);
            vert[3] = face.point + Float3(0, 0.5f, -0.5f);
        }
        else if (face.axis == Axis::nX)
        {
            vert[0] = face.point + Float3(0, 0.5f, 0.5f);
            vert[1] = face.point + Float3(0, 0.5f, -0.5f);
            vert[2] = face.point + Float3(0, -0.5f, -0.5f);
            vert[3] = face.point + Float3(0, -0.5f, 0.5f);           
        }
        else if (face.axis == Axis::pZ)
        {           
            vert[0] = face.point + Float3(0.5f, 0.5f, 0);
            vert[1] = face.point + Float3(-0.5f, 0.5f, 0);
            vert[2] = face.point + Float3(-0.5f, -0.5f, 0);
            vert[3] = face.point + Float3(0.5f, -0.5f, 0);        
        }
        else if (face.axis == Axis::nZ)
        {   
            vert[0] = face.point + Float3(0.5f, 0.5f, 0);
            vert[1] = face.point + Float3(0.5f, -0.5f, 0);
            vert[2] = face.point + Float3(-0.5f, -0.5f, 0);
            vert[3] = face.point + Float3(-0.5f, 0.5f, 0);
        }

        std::vector<size_t> polyIdx(4);

        for (int i = 0; i < 4; ++i)
        {
            if (vertsMap.find(vert[i]) != vertsMap.end())
            {
                polyIdx[i] = vertsMap[vert[i]];
            }
            else
            {
                polyIdx[i] = idx++;
                vertsMap[vert[i]] = polyIdx[i];

                verts.push_back(vert[i]);
            }
        }

        polygons.push_back(polyIdx);
    }

    Halfedge_Mesh mesh;
    
    std::cout << mesh.from_poly(polygons, verts) << "\n";

    //mesh.subdivide(SubD::linear);
    //mesh.subdivide(SubD::linear);
    //mesh.subdivide(SubD::catmullclark);
    //mesh.subdivide(SubD::catmullclark);
    //mesh.subdivide(SubD::catmullclark);

    auto triangles = std::make_shared<std::vector<Triangle>>();

    mesh.to_triangles(*triangles);

    return triangles;
}

void MarchingCubeMeshGenerator::Init()
{
    marchingCubes.resize((uint)MarchingCubeType::count);
    LoadScene("resources/models/roundcubes/1.obj", marchingCubes[0]);
    MeshScale(marchingCubes[0], marchingCubes[0], 0.5f);

    MeshMirror(marchingCubes[1], marchingCubes[0], Axis::pX);
    MeshMirror(marchingCubes[2], marchingCubes[0], Axis::pY);
    MeshMirror(marchingCubes[3], marchingCubes[1], Axis::pY);

    MeshMirror(marchingCubes[4], marchingCubes[0], Axis::pZ);
    MeshMirror(marchingCubes[5], marchingCubes[1], Axis::pZ);
    MeshMirror(marchingCubes[6], marchingCubes[2], Axis::pZ);
    MeshMirror(marchingCubes[7], marchingCubes[3], Axis::pZ);
}

std::shared_ptr<std::vector<Triangle>> MarchingCubeMeshGenerator::VoxelToMesh()
{
    auto triangles = std::make_shared<std::vector<Triangle>>();

    for (uint i = 0; i < voxels.kMapDim + 1; ++i)
    {
        for (uint j = 0; j < voxels.kMapDim + 1; ++j)
        {
            for (uint k = 0; k < voxels.kMapDimY + 1; ++k)
            {
                std::vector<uint> blocks = voxels.GetNeighborBlockAt2(i, k, j);

                uint blockCount = 0;
                for (int m = 0; m < 8; ++m)
                    blockCount += isSolid(blocks[m]);

                if (blockCount == 1)
                {
                    for (int m = 0; m < 8; ++m)
                    {
                        if (isSolid(blocks[m]))
                        {
                            std::vector<Triangle> currentMarchingCube;
                            MeshTraslate(currentMarchingCube, marchingCubes[m], Float3(i, k, j));
                            triangles->insert(triangles->end(), currentMarchingCube.begin(), currentMarchingCube.end());
                        }
                    }   
                }
            }
        }
    }

	return triangles;
}