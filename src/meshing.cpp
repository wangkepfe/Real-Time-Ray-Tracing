#include "meshing.h"
#include "terrain.h"
#include "mesh.h"
#include "fileUtils.cuh"
#include <unordered_map>
#include <iostream>

namespace
{

using namespace std;

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

template<typename... T>
inline bool isSolid(uint block, T... terms)
{
    return isSolid(block) && isSolid(terms...);
}

inline bool isSolid(const std::vector<uint>& blocks, const std::vector<int>& idx)
{
    bool result = true;
    for (int i : idx)
        result = result && isSolid(blocks[i]);
    return result;
}

// right hand coordinate system

//  [5]y  [4]
//[1]  |[0]
//     o - x
//    /
//   z
//  [7]   [6]
//[3]   [2]
enum class MarchingCubeType
{
    //   0 y   0
    // 0   | 1
    //     o - x
    //    /
    //   z
    //   0     0
    // 0     0
    t1_px_py_pz,
    t1_nx_py_pz,
    t1_px_ny_pz,
    t1_nx_ny_pz,
    t1_px_py_nz,
    t1_nx_py_nz,
    t1_px_ny_nz,
    t1_nx_ny_nz,

    //   0 y   1
    // 0   | 1
    //     o - x
    //    /
    //   z
    //   0     0
    // 0     0
    t2_dz_px_py,
    t2_dz_px_ny,
    t2_dz_nx_py,
    t2_dz_nx_ny,
    t2_dx_pz_py,
    t2_dx_pz_ny,
    t2_dx_nz_py,
    t2_dx_nz_ny,
    t2_dy_px_pz,
    t2_dy_px_nz,
    t2_dy_nx_pz,
    t2_dy_nx_nz,

    //   0 y   0
    // 0   | 1
    //     o - x
    //    /
    //   z
    //   0     1
    // 0     0
    t3_0_6, t3_2_4, t3_1_7, t3_3_5,
    t3_1_4, t3_0_5, t3_3_6, t3_2_7,

    //   0 y   0
    // 0   | 1
    //     o - x
    //    /
    //   z
    //   1     0
    // 0     0
    t10_0_7, t10_1_6, t10_4_3, t10_5_2,

    //  [5]y  [4]
    //[1]  |[0]
    //     o - x
    //    /
    //   z
    //  [7]   [6]
    //[3]   [2]

    //   0 y   1
    // 0   | 1
    //     o - x
    //    /
    //   z
    //   0     0
    // 0     1
    
    count,
};

inline Float3 PointRotate(const Float3& v, Axis axis, int angle)
{
    constexpr int cos90 = 0;
    constexpr int sin90 = 1;
    constexpr int cosneg90 = 0;
    constexpr int sinneg90 = -1;
    constexpr int cos180 = -1;
    constexpr int sin180 = 0;

    int cosTheta;
    int sinTheta;

    if (angle == 90) 
    {
        cosTheta = cos90;
        sinTheta = sin90;
    }
    else if (angle == -90) 
    {
        cosTheta = cosneg90;
        sinTheta = sinneg90;
    }
    else if (angle == 180) 
    {
        cosTheta = cos180;
        sinTheta = sin180;
    }
    else
    {
        cosTheta = 1;
        sinTheta = 0;
    }

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

inline void MeshRotate(std::vector<Triangle>& out, const std::vector<Triangle>& in, Axis axis, int angle)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
    {
        out[i].v1 = PointRotate(in[i].v1, axis, angle);
        out[i].v2 = PointRotate(in[i].v2, axis, angle);
        out[i].v3 = PointRotate(in[i].v3, axis, angle);  
    }
}

inline void PointListRotate(std::vector<Float3>& out, const std::vector<Float3>& in, Axis axis, int angle)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
        out[i] = PointRotate(in[i], axis, angle);
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

inline void PointListMirror(std::vector<Float3>& out, const std::vector<Float3>& in, Axis axis)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
        out[i] = PointMirror(in[i], axis);
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

void PointsToIndices(vector<int>& idx, const vector<Float3>& v)
{
    for (int i = 0; i < v.size(); ++i)
    {
        idx[i] = signbit(v[i].x) + signbit(v[i].y) * 2 + signbit(v[i].z) * 4;
    }
}

void MarchingCubeHelper(
    vector<vector<Triangle>>& meshes, 
    vector<vector<int>>& idx, 
    const vector<Transformation>& transList, 
    vector<vector<Float3>>& v, 
    int& meshOffset) 
{
    for (int i = 0; i < transList.size(); ++i)
    {
		const auto& trans = transList[i];

        if (trans.type == Transformation::Mirror)
        {
            PointListMirror(v[i], v[trans.basedOnId], trans.axis);
            MeshMirror(meshes[meshOffset + i], meshes[meshOffset + trans.basedOnId], trans.axis);
        }
        else if (trans.type == Transformation::Rotate)
        {
            PointListRotate(v[i], v[trans.basedOnId], trans.axis, trans.angle);
            MeshRotate(meshes[meshOffset + i], meshes[meshOffset + trans.basedOnId], trans.axis, trans.angle);
        }

        PointsToIndices(idx[i], v[i]);
    }
}

void MarchingCube3(
    vector<vector<Triangle>>& marchingCubes,
    vector<vector<int>>& idx,
    int& meshOffset)
{
    LoadScene("resources/models/roundcubes/4.obj", marchingCubes[meshOffset]);
    MeshScale(marchingCubes[meshOffset], marchingCubes[meshOffset], 0.5f);

    idx.resize(24);
    for (auto& id : idx)
        id.resize(3);

    vector<vector<Float3>> v(24, vector<Float3>(3));
    v[0][0] = Float3(1,1,1);
    v[0][1] = Float3(1,-1,1);
    v[0][2] = Float3(1,1,-1);

    vector<Transformation> transList = {
       {Transformation::None, Axis::pX, 0, 0},
       {Transformation::Rotate, Axis::pX, 90, 0}, 
       {Transformation::Rotate, Axis::pX, 180, 0}, 
       {Transformation::Rotate, Axis::pX, -90, 0},
		
       {Transformation::Rotate, Axis::pY, 90, 0},
       {Transformation::Rotate, Axis::pY, 90, 1},
       {Transformation::Rotate, Axis::pY, 90, 2},
       {Transformation::Rotate, Axis::pY, 90, 3},
		
       {Transformation::Rotate, Axis::pY, 180, 0},
       {Transformation::Rotate, Axis::pY, 180, 1},
       {Transformation::Rotate, Axis::pY, 180, 2},
       {Transformation::Rotate, Axis::pY, 180, 3},
		
       {Transformation::Rotate, Axis::pY, -90, 0},
       {Transformation::Rotate, Axis::pY, -90, 1},
       {Transformation::Rotate, Axis::pY, -90, 2},
       {Transformation::Rotate, Axis::pY, -90, 3},
		
       {Transformation::Rotate, Axis::pZ, 90, 0},
       {Transformation::Rotate, Axis::pZ, 90, 1},
       {Transformation::Rotate, Axis::pZ, 90, 2},
       {Transformation::Rotate, Axis::pZ, 90, 3},

       {Transformation::Rotate, Axis::pZ, -90, 0},
       {Transformation::Rotate, Axis::pZ, -90, 1},
       {Transformation::Rotate, Axis::pZ, -90, 2},
       {Transformation::Rotate, Axis::pZ, -90, 3},
    };

    MarchingCubeHelper(marchingCubes, idx, transList, v, meshOffset);
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
    marchingCubes.resize((uint)MarchingCubeType::count + 24);
    idx.resize(3);

    int i = 0;

    // t1
    {
        LoadScene("resources/models/roundcubes/1.obj", marchingCubes[0]);
        MeshScale(marchingCubes[0], marchingCubes[0], 0.5f);

        MeshMirror(marchingCubes[1], marchingCubes[0], Axis::pX);
        MeshMirror(marchingCubes[2], marchingCubes[0], Axis::pY);
        MeshMirror(marchingCubes[3], marchingCubes[1], Axis::pY);

        MeshMirror(marchingCubes[4], marchingCubes[0], Axis::pZ);
        MeshMirror(marchingCubes[5], marchingCubes[1], Axis::pZ);
        MeshMirror(marchingCubes[6], marchingCubes[2], Axis::pZ);
        MeshMirror(marchingCubes[7], marchingCubes[3], Axis::pZ);

        i += 8;
    }

    // t2
    {
        LoadScene("resources/models/roundcubes/2.obj", marchingCubes[i]);
        MeshScale(marchingCubes[i], marchingCubes[i], 0.5f);

        MeshMirror(marchingCubes[i + 1], marchingCubes[i + 0], Axis::pY);
        MeshMirror(marchingCubes[i + 2], marchingCubes[i + 0], Axis::pX);
        MeshMirror(marchingCubes[i + 3], marchingCubes[i + 1], Axis::pX);

        MeshRotate(marchingCubes[i + 4], marchingCubes[i + 0], Axis::pY, -90);
        MeshRotate(marchingCubes[i + 5], marchingCubes[i + 1], Axis::pY, -90);
        MeshRotate(marchingCubes[i + 6], marchingCubes[i + 2], Axis::pY, -90);
        MeshRotate(marchingCubes[i + 7], marchingCubes[i + 3], Axis::pY, -90);

        MeshRotate(marchingCubes[i + 8],  marchingCubes[i + 0], Axis::pX, 90);
        MeshRotate(marchingCubes[i + 9],  marchingCubes[i + 1], Axis::pX, 90);
        MeshRotate(marchingCubes[i + 10], marchingCubes[i + 2], Axis::pX, 90);
        MeshRotate(marchingCubes[i + 11], marchingCubes[i + 3], Axis::pX, 90);

        i += 12;
    }

    // t3
    {
        LoadScene("resources/models/roundcubes/3.obj", marchingCubes[i]);
        MeshScale(marchingCubes[i], marchingCubes[i], 0.5f);

        MeshMirror(marchingCubes[i + 1], marchingCubes[i + 0], Axis::pY);
        MeshMirror(marchingCubes[i + 2], marchingCubes[i + 0], Axis::pX);
        MeshMirror(marchingCubes[i + 3], marchingCubes[i + 2], Axis::pY);

        MeshRotate(marchingCubes[i + 4], marchingCubes[i + 0], Axis::pZ, 90);
        MeshRotate(marchingCubes[i + 5], marchingCubes[i + 1], Axis::pZ, 90);
        MeshRotate(marchingCubes[i + 6], marchingCubes[i + 2], Axis::pZ, 90);
        MeshRotate(marchingCubes[i + 7], marchingCubes[i + 3], Axis::pZ, 90);

        i += 8;
    }

    // t10
    {
        LoadScene("resources/models/roundcubes/10.obj", marchingCubes[i]);
        MeshScale(marchingCubes[i], marchingCubes[i], 0.5f);

        MeshMirror(marchingCubes[i + 1], marchingCubes[i + 0], Axis::pX);
        MeshMirror(marchingCubes[i + 2], marchingCubes[i + 0], Axis::pZ);
        MeshMirror(marchingCubes[i + 3], marchingCubes[i + 1], Axis::pZ);

        i += 4;
    }

    MarchingCube3(marchingCubes, idx[2], i);
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
                // (), (-x), (-y), (-x, -y), (-z), (-x, -z), (-y, -z), (-x,-y, -z)
                //
                //  [5]y  [4]
                //[1]  |[0]
                //     o - x
                //    /
                //   z
                //  [7]   [6]
                //[3]   [2]
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
                else if (blockCount == 2)
                {
                    bool matchingSequence[24] = {
                        isSolid(blocks[0], blocks[4]), isSolid(blocks[2], blocks[6]), isSolid(blocks[1], blocks[5]), isSolid(blocks[3], blocks[7]),
                        isSolid(blocks[0], blocks[1]), isSolid(blocks[2], blocks[3]), isSolid(blocks[4], blocks[5]), isSolid(blocks[6], blocks[7]),
                        isSolid(blocks[0], blocks[2]), isSolid(blocks[4], blocks[6]), isSolid(blocks[1], blocks[3]), isSolid(blocks[5], blocks[7]),

                        isSolid(blocks[0], blocks[6]), isSolid(blocks[2], blocks[4]), isSolid(blocks[1], blocks[7]), isSolid(blocks[3], blocks[5]),
                        isSolid(blocks[1], blocks[4]), isSolid(blocks[0], blocks[5]), isSolid(blocks[3], blocks[6]), isSolid(blocks[2], blocks[7]),

                        isSolid(blocks[0], blocks[7]), isSolid(blocks[1], blocks[6]), isSolid(blocks[4], blocks[3]), isSolid(blocks[5], blocks[2]),
                    };
                    for (int m = 0; m < 24; ++m)
                    {
                        if (matchingSequence[m])
                        {
                            std::vector<Triangle> currentMarchingCube;
                            MeshTraslate(currentMarchingCube, marchingCubes[m + 8], Float3(i, k, j));
                            triangles->insert(triangles->end(), currentMarchingCube.begin(), currentMarchingCube.end());
                        } 
                    }
                }
                else if (blockCount == 3)
                {
                    for (int m = 0; m < 24; ++m)
                    {
                        if (isSolid(blocks, idx[2][m]))
                        {
                            std::vector<Triangle> currentMarchingCube;
                            MeshTraslate(currentMarchingCube, marchingCubes[m + 8 + 24], Float3(i, k, j));
                            triangles->insert(triangles->end(), currentMarchingCube.begin(), currentMarchingCube.end());
                        }
                    }
                }
            }
        }
    }

	return triangles;
}