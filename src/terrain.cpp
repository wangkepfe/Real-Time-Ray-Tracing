#include "terrain.h"
#include "perlin.h"
#include <iostream>

void Chunk::Generate(const Perlin& perlin, float noiseScale, float baseY, float scaleY)
{
    for (uint i = 0; i < kBlockDim; ++i)
    {
        for (uint j = 0; j < kBlockDim; ++j)
        {
            float nx = (float)(x * kBlockDim + i);
			float nz = (float)(z * kBlockDim + j);

            nx *= noiseScale / (float)kBlockDim;
            nz *= noiseScale / (float)kBlockDim;

            float noiseVal = perlin.noise3D(nx, nz, 0.8f);
            noiseVal -= 0.5f;
            noiseVal = baseY + noiseVal * scaleY;

            for (uint k = 0; k < kBlockDimY; ++k)
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

void VoxelsGenerator::Generate()
{
    for (uint i = 0; i < kChunkDim; ++i)
    {
        for (uint j = 0; j < kChunkDim; ++j)
        {
			chunks[i][j].x = i;
			chunks[i][j].z = j;
            chunks[i][j].Generate(perlin, noiseScale, baseY, scaleY);
        }
    }
}

uint VoxelsGenerator::GetBlockAt(uint x, uint y, uint z) const
{
    uint cx = x / Chunk::kBlockDim;
    x = x % Chunk::kBlockDim;

    uint cz = z / Chunk::kBlockDim;
    z = z % Chunk::kBlockDim;

    return chunks[cx][cz].blocks[y][x][z];
}

std::vector<uint> VoxelsGenerator::GetNeighborBlockAt(uint x, uint y, uint z) const
{
    uint cx = x / Chunk::kBlockDim;
    x = x % Chunk::kBlockDim;

    uint cz = z / Chunk::kBlockDim;
    z = z % Chunk::kBlockDim;

    std::vector<uint> result;

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