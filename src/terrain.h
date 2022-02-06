#pragma once

#include <vector>
#include <unordered_set>
#include <memory>
#include "linear_math.h"
#include "geometry.h"
#include "perlin.h"

class Chunk
{
public:
    Chunk() {}
    ~Chunk() {}

    void Generate(const Perlin& perlin, float scale, float baseY, float scaleY);

    static const uint kBlockDim = 16;
    static const uint kBlockDimY = 16;

    uint x, z;
    uint blocks[kBlockDimY][kBlockDim][kBlockDim] = {}; // y x z, horizontal slices
};

class VoxelsGenerator
{
public:
    VoxelsGenerator() {}
    ~VoxelsGenerator() {}

    void Generate();

    uint GetBlockAt(uint x, uint y, uint z) const;
    std::vector<uint> GetNeighborBlockAt(uint x, uint y, uint z) const;
    std::vector<uint> GetNeighborBlockAt2(uint x, uint y, uint z) const;

    float noiseScale = 2.0f;
    float meshScale = 0.5f;
    float baseY = Chunk::kBlockDimY / 2.0f;
    float scaleY = Chunk::kBlockDimY / 2.0f;

    static const uint kChunkDim = 1;
    static const uint kMapDim = kChunkDim * Chunk::kBlockDim;
    static const uint kMapDimY = Chunk::kBlockDimY;

private:
    Chunk chunks[kChunkDim][kChunkDim];
    Perlin perlin;
};