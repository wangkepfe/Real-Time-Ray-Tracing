#include "terrain.h"

#include "perlin.h"
#include "mesh.h"

#include <iostream>
#include <unordered_set>
#include <unordered_map>

void Chunk::Generate(const Perlin& perlin, float noiseScale, float baseY, float scaleY)
{
    for (ushort i = 0; i < kBlockDim; ++i)
    {
        for (ushort j = 0; j < kBlockDim; ++j)
        {
            float nx = (float)(x * kBlockDim + i);
			float nz = (float)(z * kBlockDim + j);

            nx *= noiseScale / (float)kBlockDim;
            nz *= noiseScale / (float)kBlockDim;

            float noiseVal = perlin.noise3D(nx, nz, 0.8f);
            noiseVal -= 0.5f;
            noiseVal = baseY + noiseVal * scaleY;

            for (ushort k = 0; k < kBlockDimY; ++k)
            {   
                if (k < noiseVal)
                {
                    blocks[k][i][j] = 1;
                }
                else
                {
                    break;
                }
            }
        }
    }
}

void TerrainGenerator::FindSurfaceFaces()
{
    for (ushort i = 0; i < kMapDim; ++i)
    {
        for (ushort j = 0; j < kMapDim; ++j)
        {
            for (ushort k = 0; k < Chunk::kBlockDimY; ++k)
            {
                ushort block = GetBlockAt(i, k, j);
                
                if (block == 1)
                {
					std::vector<ushort> neighbors = GetNeighborBlockAt(i, k, j);

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
}

void TerrainGenerator::SurfaceToTriangles()
{
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

    mesh.to_triangles(triangles);
}

void TerrainGenerator::FillingChunks()
{
    for (ushort i = 0; i < kChunkDim; ++i)
    {
        for (ushort j = 0; j < kChunkDim; ++j)
        {
			chunks[i][j].x = i;
			chunks[i][j].z = j;
            chunks[i][j].Generate(perlin, noiseScale, baseY, scaleY);
        }
    }
}

void TerrainGenerator::Generate()
{
    FillingChunks();
    FindSurfaceFaces();
    SurfaceToTriangles();
}

ushort TerrainGenerator::GetBlockAt(ushort x, ushort y, ushort z) const
{
    ushort cx = x / Chunk::kBlockDim;
    x = x % Chunk::kBlockDim;

    ushort cz = z / Chunk::kBlockDim;
    z = z % Chunk::kBlockDim;

    return chunks[cx][cz].blocks[y][x][z];
}

std::vector<ushort> TerrainGenerator::GetNeighborBlockAt(ushort x, ushort y, ushort z) const
{
    ushort cx = x / Chunk::kBlockDim;
    x = x % Chunk::kBlockDim;

    ushort cz = z / Chunk::kBlockDim;
    z = z % Chunk::kBlockDim;

    std::vector<ushort> result;

    // y-
    if (y == 0)
    {
        result.push_back(0xFFFF);
    }
    else
    {
        result.push_back(chunks[cx][cz].blocks[y - 1][x][z]);
    }

    // y+
    if (y == Chunk::kBlockDimY - 1)
    {
        result.push_back(0xFFFF);
    }
    else
    {
        result.push_back(chunks[cx][cz].blocks[y + 1][x][z]);
    }

    // x-
    if (x == 0 && cx == 0) 
    {
        result.push_back(0xFFFF);
    }
    else if (x == 0)
    {
        result.push_back(chunks[cx - 1][cz].blocks[y][Chunk::kBlockDim - 1][z]);
    }
    else
    {
        result.push_back(chunks[cx][cz].blocks[y][x - 1][z]);
    }

    // x+
    if (x == Chunk::kBlockDim - 1 && cx == kChunkDim - 1)
    {
        result.push_back(0xFFFF);
    }
    else if (x == Chunk::kBlockDim - 1)
    {
        result.push_back(chunks[cx + 1][cz].blocks[y][0][z]);
    }
    else
    {
        result.push_back(chunks[cx][cz].blocks[y][x + 1][z]);
    }

    // z-
    if (z == 0 && cz == 0) 
    {
        result.push_back(0xFFFF);
    }
    else if (z == 0)
    {
        result.push_back(chunks[cx][cz - 1].blocks[y][x][Chunk::kBlockDim - 1]);
    }
    else
    {
        result.push_back(chunks[cx][cz].blocks[y][x][z - 1]);
    }

    // z+
    if (z == Chunk::kBlockDim - 1 && cz == kChunkDim - 1)
    {
        result.push_back(0xFFFF);
    }
    else if (z == Chunk::kBlockDim - 1)
    {
        result.push_back(chunks[cx][cz + 1].blocks[y][x][0]);
    }
    else
    {
        result.push_back(chunks[cx][cz].blocks[y][x][z + 1]);
    }

    return result;
}