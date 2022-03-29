#pragma once

#include "linearMath.h"

enum MipmapTextureName
{
	SoilAlbedoAo,
	SoilNormalRoughness,
	SoilHeight,

	MipmapTextureCount,
};

struct Mipmap
{
    static constexpr uint numMipLevels = 11;
    cudaSurfaceObject_t mip[numMipLevels];
    Int2 size[numMipLevels];
    int maxLod;
};

struct TextureAtlas
{
	Mipmap mipmapTex[MipmapTextureCount];
};