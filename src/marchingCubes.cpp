#include "meshing.h"
#include "terrain.h"
#include "mesh.h"
#include "fileUtils.cuh"
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <utility>
#include <memory>

namespace {

inline bool IsBorder(uint block)
{
	return block == 0xFFFF;
}

inline bool IsEmpty(uint block)
{
    return block == 0;
}

inline uint ToggleSingleBit(uint i, uint bit)
{
    return i ^ (1UL << bit);
}

inline uint FlipBits(uint n, uint k) {
    uint mask = (1UL << k) - 1;
    return ~n & mask;
}

//  [5]y  [4]
//[1]  |[0]
//     o - x
//    /
//   z
//  [7]   [6]
//[3]   [2]
inline bool IsSolid(uint i, const std::vector<uint>& blocks)
{
    if (IsEmpty(blocks[i]))
        return false;

    if (!IsBorder(blocks[i]))
        return true;

    uint neighborX = ToggleSingleBit(i, 0);
    uint neighborY = ToggleSingleBit(i, 1);
    uint neighborZ = ToggleSingleBit(i, 2);

    uint neighborXY = ToggleSingleBit(neighborX, 1);
    uint neighborXZ = ToggleSingleBit(neighborX, 2);
    uint neighborYZ = ToggleSingleBit(neighborY, 2);

    uint oppositeCorner = FlipBits(i, 3);

    bool isNeighborXBorder = IsBorder(blocks[neighborX]);
    bool isNeighborYBorder = IsBorder(blocks[neighborY]);
    bool isNeighborZBorder = IsBorder(blocks[neighborZ]);

    bool isNeighborXYBorder = IsBorder(blocks[neighborXY]);
    bool isNeighborXZBorder = IsBorder(blocks[neighborXZ]);
    bool isNeighborYZBorder = IsBorder(blocks[neighborYZ]);

	bool onWallX = isNeighborYBorder && isNeighborZBorder && isNeighborYZBorder;
	bool onWallY = isNeighborXBorder && isNeighborZBorder && isNeighborXZBorder;
	bool onWallZ = isNeighborXBorder && isNeighborYBorder && isNeighborXYBorder;

    if (onWallX && onWallY && onWallZ)
        return !IsEmpty(blocks[oppositeCorner]);

	else if (onWallY && onWallZ)
		return !IsEmpty(blocks[neighborYZ]);

	else if (onWallX && onWallZ)
		return !IsEmpty(blocks[neighborXZ]);

	else if (onWallX && onWallY)
		return !IsEmpty(blocks[neighborXY]);

    else if (onWallZ)
        return !IsEmpty(blocks[neighborZ]);

    else if (onWallY)
        return !IsEmpty(blocks[neighborY]);

    else if (onWallX)
        return !IsEmpty(blocks[neighborX]);
}

inline uint BlocksToIdx(const std::vector<uint>& blocks)
{
    uint result = 0;
    for (int i = 0; i < 8; ++i)
    {
        result += IsSolid(i, blocks) * (1u << i);
    }
    return result;
}

inline uint PointsToIdx(const std::vector<Float3>& points)
{
    uint result = 0;
    for (int i = 0; i < points.size(); ++i)
    {
        uint quadrant = signbit(points[i].x) + signbit(points[i].y) * 2u + signbit(points[i].z) * 4u;
        result += (1u << quadrant);
    }
    return result;
}

inline Float3 PointRotate(const Float3& v, Axis axis, int angle)
{
    int cosTheta, sinTheta;

    if      (angle == 90)  { cosTheta = 0; sinTheta = 1;  }
    else if (angle == -90) { cosTheta = 0; sinTheta = -1; }
    else if (angle == 180) { cosTheta = -1; sinTheta = 0; }
    else                   { cosTheta = 1; sinTheta = 0;  }

    if      (axis == Axis::pY) { return Float3(cosTheta * v.x + sinTheta * v.z , v.y, - sinTheta * v.x + cosTheta * v.z); }
    else if (axis == Axis::pX) { return Float3(v.x, cosTheta * v.y - sinTheta * v.z, sinTheta * v.y + cosTheta * v.z); }
    else if (axis == Axis::pZ) { return Float3(cosTheta * v.x - sinTheta * v.y , sinTheta * v.x + cosTheta * v.y, v.z); }
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

        out[i].n1 = PointRotate(in[i].n1, axis, angle);
        out[i].n2 = PointRotate(in[i].n2, axis, angle);
        out[i].n3 = PointRotate(in[i].n3, axis, angle);
    }
}

inline void PointListRotate(std::vector<Float3>& out, const std::vector<Float3>& in, Axis axis, int angle)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
        out[i] = PointRotate(in[i], axis, angle);
}

inline void MeshCopy(std::vector<Triangle>& out, const std::vector<Triangle>& in)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
    {
        out[i].v1 = in[i].v1;
        out[i].v2 = in[i].v2;
        out[i].v3 = in[i].v3;
        out[i].n1 = in[i].n1;
        out[i].n2 = in[i].n2;
        out[i].n3 = in[i].n3;
    }
}

inline void MeshFlipNormal(std::vector<Triangle>& out, const std::vector<Triangle>& in)
{
    int s = in.size();
    out.resize(s);
    for (int i = 0; i < s; ++i)
    {
        out[i].v1 = in[i].v1;
        out[i].v2 = in[i].v3;
        out[i].v3 = in[i].v2;

        out[i].n1 = -in[i].n1;
        out[i].n2 = -in[i].n3;
        out[i].n3 = -in[i].n2;
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
        out[i].n1 = in[i].n1;
        out[i].n2 = in[i].n2;
        out[i].n3 = in[i].n3;
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
        out[i].n1 = in[i].n1;
        out[i].n2 = in[i].n2;
        out[i].n3 = in[i].n3;
    }
}

} // namespace

void MarchingCubeMeshGenerator::InitMarchingCube(const MarchingCube& marchingCube)
{
    std::vector<uint> meshIdx(transList.size());
    meshIdx[0] = PointsToIdx(marchingCube.points);

    int subdivisionLevel = 2;

    std::string filename = "resources/models/roundcubes/" + std::to_string(subdivisionLevel) + "/" + marchingCube.filename;

    LoadScene(filename.c_str(), meshes[meshIdx[0]]);

    MeshScale(meshes[meshIdx[0]], meshes[meshIdx[0]], 0.5f); // @TODO: remove this

	if (marchingCube.reversible)
	{
		MeshFlipNormal(meshes[FlipBits(meshIdx[0], 8u)], meshes[meshIdx[0]]);
	}

    std::vector<std::vector<Float3>> points(transList.size());
    points[0] = marchingCube.points;

    for (int i = 1; i < transList.size(); ++i)
    {
		const auto& trans = transList[i];

        PointListRotate(points[i], points[trans.basedOnId], trans.axis, trans.angle);

        meshIdx[i] = PointsToIdx(points[i]);

		uint srcIdx = meshIdx[trans.basedOnId];
		uint destIdx = meshIdx[i];

        if (meshes[destIdx].size() == 0)
        {
			// std::cout << destIdx << " ";
            MeshRotate(meshes[destIdx], meshes[srcIdx], trans.axis, trans.angle);
        }

		srcIdx = meshIdx[i];
		destIdx = FlipBits(meshIdx[i], 8u);

        if (marchingCube.reversible && meshes[destIdx].size() == 0)
        {
			// std::cout << destIdx << " ";
            MeshFlipNormal(meshes[destIdx], meshes[srcIdx]);
        }
    }
	std::cout << "\n\n";
}

void MarchingCubeMeshGenerator::Init()
{
    meshes.resize(256);

    transList = std::move(std::vector<Transformation> {
        {Transformation::None  , Axis::pX, 0  , 0},
        {Transformation::Rotate, Axis::pX, 90 , 0},
        {Transformation::Rotate, Axis::pX, 180, 0},
        {Transformation::Rotate, Axis::pX, -90, 0},
        {Transformation::Rotate, Axis::pY, 90 , 0},
        {Transformation::Rotate, Axis::pY, 90 , 1},
        {Transformation::Rotate, Axis::pY, 90 , 2},
        {Transformation::Rotate, Axis::pY, 90 , 3},
        {Transformation::Rotate, Axis::pY, 180, 0},
        {Transformation::Rotate, Axis::pY, 180, 1},
        {Transformation::Rotate, Axis::pY, 180, 2},
        {Transformation::Rotate, Axis::pY, 180, 3},
        {Transformation::Rotate, Axis::pY, -90, 0},
        {Transformation::Rotate, Axis::pY, -90, 1},
        {Transformation::Rotate, Axis::pY, -90, 2},
        {Transformation::Rotate, Axis::pY, -90, 3},
        {Transformation::Rotate, Axis::pZ, 90 , 0},
        {Transformation::Rotate, Axis::pZ, 90 , 1},
        {Transformation::Rotate, Axis::pZ, 90 , 2},
        {Transformation::Rotate, Axis::pZ, 90 , 3},
        {Transformation::Rotate, Axis::pZ, -90, 0},
        {Transformation::Rotate, Axis::pZ, -90, 1},
        {Transformation::Rotate, Axis::pZ, -90, 2},
        {Transformation::Rotate, Axis::pZ, -90, 3},
    });

    std::vector<MarchingCube> marchingCubes =
    {
        //   0 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   0     0
        // 0     0
        {
            "1.obj",
            {
                Float3(1,1,1),
            },
            true,
        },
        //   0 y   1
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   0     0
        // 0     0
        {
            "2.obj",
            {
                Float3(1,1,1),
                Float3(1,1,-1),
            },
            true,
        },
        //   0 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   0     1
        // 0     0
        {
            "3.obj",
            {
                Float3(1,1,1),
                Float3(1,-1,-1),
            },
        },
        //   0 y   1
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   0     0
        // 0     1
        {
            "4.obj",
            {
                Float3(1,1,1),
                Float3(1,-1,1),
                Float3(1,1,-1),
            },
            true,
        },
        //   1 y   1
        // 1   | 1
        //     o - x
        //    /
        //   z
        //   0     0
        // 0     0
        {
            "5.obj",
            {
                Float3(1,1,1),
                Float3(1,1,-1),
                Float3(-1,1,1),
                Float3(-1,1,-1),
            },
        },
        //   0 y   1
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   1     0
        // 0     1
        {
            "6.obj",
            {
				Float3(1,1,1),
				Float3(1,-1,1),
				Float3(1,1,-1),
                Float3(-1,-1,-1),
            },
        },
        //   1 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   0     1
        // 1     0
        {
            "7.obj",
            {
				Float3(1,1,1),
				Float3(1,-1,-1),
				Float3(-1,1,-1),
                Float3(-1,-1,1),
            },
        },
        //   0 y   1
        // 1   | 1
        //     o - x
        //    /
        //   z
        //   0     0
        // 0     1
        {
            "8.obj",
            {
                Float3(1,1,1),
                Float3(1,-1,1),
                Float3(1,1,-1),
                Float3(-1,1,1),
            },
        },
        //   1 y   0
        // 1   | 1
        //     o - x
        //    /
        //   z
        //   1     0
        // 0     0
        {
            "9.obj",
            {
                Float3(1,1,1),
                Float3(-1,1,1),
                Float3(-1,1,-1),
                Float3(-1,-1,-1),
            },
        },
        //   0 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   1     0
        // 0     0
        {
            "10.obj",
            {
                Float3(1,1,1),
                Float3(-1,-1,-1),
            },
            true,
        },
        //   0 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   1     0
        // 1     0
        {
            "11.obj",
            {
                Float3(1,1,1),
                Float3(-1,-1,1),
                Float3(-1,-1,-1),
            },
        },
        //   1 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   0     0
        // 1     0
        {
            "12.obj",
            {
                Float3(1,1,1),
                Float3(-1,1,-1),
                Float3(-1,-1,1),
            },
        },
        //   1 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   1     0
        // 0     1
        {
            "13.obj",
            {
                Float3(1,1,1),
                Float3(1,-1,1),
                Float3(-1,1,-1),
                Float3(-1,-1,-1),
            },
        },
        //   0 y   0
        // 0   | 1
        //     o - x
        //    /
        //   z
        //   1     1
        // 0     1
        {
            "14.obj",
            {
                Float3(1,1,1),
                Float3(1,-1,1),
                Float3(1,-1,-1),
                Float3(-1,-1,-1),
            },
        },
        //   1 y   1
        // 1   | 0
        //     o - x
        //    /
        //   z
        //   1     0
        // 1     1
        {
            "15.obj",
            {
                Float3(1,1,-1),
                Float3(1,-1,1),
                Float3(-1,1,1),
                Float3(-1,1,-1),
                Float3(-1,-1,1),
                Float3(-1,-1,-1),
            },
        },
    };

    for (const auto& marchingCube : marchingCubes)
        InitMarchingCube(marchingCube);
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
                uint id = BlocksToIdx(blocks);
                std::vector<Triangle> currentMarchingCube;
                MeshTraslate(currentMarchingCube, meshes[id], Float3(i, k, j));
                triangles->insert(triangles->end(), currentMarchingCube.begin(), currentMarchingCube.end());
            }
        }
    }

	return triangles;
}

using namespace std;

class VertexMerger
{
public:
    VertexMerger(vector<uint>& indices, vector<Float3>& vertices, Float3 sceneBoundMax, Float3 sceneBoundMin, uint vertexCount, float maxDistanceAllowed)
    :   indices            { indices            },
        vertices           { vertices           },
        sceneBoundMax      { sceneBoundMax      },
        sceneBoundMin      { sceneBoundMin      },
        vertexCount        { vertexCount        },
        maxDistanceAllowed { maxDistanceAllowed },
        bin                {                    },
        count              { 0                  },
        binSize            { 0                  },
        binDim             { 0                  }
    {}

    void Init()
    {
        Float3 sceneExtent = sceneBoundMax - sceneBoundMin;

        // a x b x c = numVertices
        // a : b : c = sceneExtent.x : sceneExtent.y : sceneExtent.z
        //
        // a x ((sceneExtent.y / sceneExtent.x) x a) x ((sceneExtent.z / sceneExtent.x) x a) = numVertices
        // a^3 x sceneExtent.y x sceneExtent.z / sceneExtent.x^2 = numVertices
        // a = (numVertices x sceneExtent.x^2 / sceneExtent.y / sceneExtent.z)^(1/3)

        float dimX = powf(vertexCount * sceneExtent.x * sceneExtent.x / sceneExtent.y / sceneExtent.z, 1.0f / 3.0f);
        float dimY = dimX / sceneExtent.x * sceneExtent.y;
        float dimZ = dimX / sceneExtent.x * sceneExtent.z;

        binDim.x = static_cast<int>(dimX) + 1;
        binDim.y = static_cast<int>(dimY) + 1;
        binDim.z = static_cast<int>(dimZ) + 1;

        binSize = sceneExtent.x / (float)binDim.x;

        bin.resize(binDim.x, vector<vector<vector<uint>>>(binDim.y, vector<vector<uint>>(binDim.z, vector<uint>{})));
    }

    void Process(const Float3& v)
    {
        uint binX;
        uint binY;
        uint binZ;

        GetBinIdx(v, binX, binY, binZ);

        for (uint i = ((binX == 0) ? binX : binX - 1); i <= ((binX == binDim.x - 1) ? binX : binX + 1); ++i)
        {
            for (uint j = ((binY == 0) ? binY : binY - 1); j <= ((binY == binDim.y - 1) ? binY : binY + 1); ++j)
            {
                for (uint k = ((binZ == 0) ? binZ : binZ - 1); k <= ((binZ == binDim.z - 1) ? binZ : binZ + 1); ++k)
                {
                    for (uint idx : bin[i][j][k])
                    {
                        if (PointMatch(v, vertices[idx]) == true)
                        {
                            // Duplicate found!
                            indices.push_back(idx);
                            return;
                        }
                    }
                }
            }
        }

        // New vertex!
        indices.push_back(count);
        vertices.push_back(v);
        bin[binX][binY][binZ].push_back(count);
        ++count;
    }

private:
    void GetBinIdx(const Float3& v, uint& binX, uint& binY, uint& binZ) const
	{
		binX = (uint)((v.x - sceneBoundMin.x) / binSize);
        binY = (uint)((v.y - sceneBoundMin.y) / binSize);
        binZ = (uint)((v.z - sceneBoundMin.z) / binSize);

        binX = std::clamp(binX, 0u, (binDim.x == 0u) ? 0u : binDim.x - 1u);
        binY = std::clamp(binY, 0u, (binDim.y == 0u) ? 0u : binDim.y - 1u);
        binZ = std::clamp(binZ, 0u, (binDim.z == 0u) ? 0u : binDim.z - 1u);
	}

    bool PointMatch(const Float3& v1, const Float3& v2) const
	{
		return distancesq(v1, v2) <= maxDistanceAllowed * maxDistanceAllowed;
	}

    vector<uint>& indices;
    vector<Float3>& vertices;
    Float3 sceneBoundMax;
    Float3 sceneBoundMin;
    uint vertexCount;
    float maxDistanceAllowed;
    vector<vector<vector<vector<uint>>>> bin;
    uint count;

    float binSize;
    UInt3 binDim;
};

struct Float3ApproxHasher
{
	unsigned long long operator() (const Float3& v) const
	{
        constexpr float binSize = 0.1f;
        constexpr unsigned long long binDim = 1u << 20;

		unsigned long long binX = (unsigned long long)(v.x / binSize) + binDim / 2;
        unsigned long long binY = (unsigned long long)(v.y / binSize) + binDim / 2;
        unsigned long long binZ = (unsigned long long)(v.z / binSize) + binDim / 2;

        binX = std::clamp(binX, 0Ui64, binDim);
        binY = std::clamp(binY, 0Ui64, binDim);
        binZ = std::clamp(binZ, 0Ui64, binDim);

		return binX + binY * binDim + binZ * binDim * binDim;
	}
};

struct Float3ApproxEqual
{
	bool operator() (const Float3& v1, const Float3& v2) const
	{
        constexpr float maxDistanceAllowed = 0.05f;
		return distancesq(v1, v2) <= maxDistanceAllowed * maxDistanceAllowed;
	}
};

void MarchingCubeMeshGenerator::VoxelToMesh(std::vector<Float3>& vertices, std::vector<uint>& indices)
{
    std::unordered_map<Float3, uint, Float3Hasher> vertsMap;

	std::vector<Float3> vertices1;
	std::vector<uint> indices1;

    Float3 sceneBoundMax = Float3(-FLT_MAX);
    Float3 sceneBoundMin = Float3(FLT_MAX);

    uint idx = 0;
    for (uint i = 0; i < voxels.kMapDim + 1; ++i)
    {
        for (uint j = 0; j < voxels.kMapDim + 1; ++j)
        {
            for (uint k = 0; k < voxels.kMapDimY + 1; ++k)
            {
                std::vector<uint> blocks = voxels.GetNeighborBlockAt2(i, k, j);
                uint id = BlocksToIdx(blocks);
                std::vector<Triangle> currentMarchingCube;
                MeshTraslate(currentMarchingCube, meshes[id], Float3(i, k, j));

                for (auto& triangle : currentMarchingCube)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        Float3 p = triangle.vertices[i].xyz;

                        if (vertsMap.find(p) == vertsMap.end())
                        {
                            vertices1.push_back(p);
                            indices1.push_back(idx);
							vertsMap[p] = idx;
                            ++idx;

                            sceneBoundMax = max3f(sceneBoundMax, p);
                            sceneBoundMin = min3f(sceneBoundMin, p);
                        }
                        else
                        {
                            indices1.push_back(vertsMap[p]);
                        }
                    }
                }
            }
        }
    }
    constexpr float maxDistanceAllowed = 0.001f;
    VertexMerger merger(indices, vertices, sceneBoundMax, sceneBoundMin, indices1.size(), maxDistanceAllowed);
    merger.Init();
    for (uint idx : indices1)
    {
        merger.Process(vertices1[idx]);
    }
}