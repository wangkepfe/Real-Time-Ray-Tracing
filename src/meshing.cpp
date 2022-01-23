#include "meshing.h"
#include "terrain.h"
#include "mesh.h"
#include <unordered_map>
#include <iostream>

enum class Axis
{
    pX, pY, pZ, nX, nY, nZ,
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