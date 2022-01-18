#pragma once

#include <vector>
#include <unordered_set>
#include "linear_math.h"
#include "geometry.h"
#include "perlin.h"

class Chunk
{
public:
    Chunk() {}
    ~Chunk() {}

    void Generate(const Perlin& perlin, float scale, float baseY, float scaleY);

    static const ushort kBlockDim = 32;
    static const ushort kBlockDimY = kBlockDim;

    ushort x, z;
    ushort blocks[kBlockDimY][kBlockDim][kBlockDim] = {}; // y x z, horizontal slices
};

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

class TerrainGenerator
{
public:
    TerrainGenerator(std::vector<Triangle>& triangles) : triangles{triangles} {}
    ~TerrainGenerator() {}

    void Generate();

    float noiseScale = 1.0f;
    float meshScale = 0.5f;
    float baseY = Chunk::kBlockDim / 2;
    float scaleY = Chunk::kBlockDim / 2;

private:
    void FillingChunks();
    void FindSurfaceFaces();
    void SurfaceToTriangles();

    ushort GetBlockAt(ushort x, ushort y, ushort z) const;
    std::vector<ushort> GetNeighborBlockAt(ushort x, ushort y, ushort z) const;

    static const ushort kChunkDim = 2;
    static const ushort kMapDim = kChunkDim * Chunk::kBlockDim;

    std::vector<Triangle>& triangles;
    std::unordered_set<QuadFace, QuadFaceHasher, QuadFaceEqualOperator> faces;

    Chunk chunks[kChunkDim][kChunkDim];
    Perlin perlin;
};