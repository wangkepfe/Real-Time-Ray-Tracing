#pragma once

#include "kernel.cuh"

template<typename VectorType>
inline  __device__ VectorType fminf(VectorType a, VectorType b);

template<>
inline __device__ float fminf<float>(float a, float b)
{
    return fminf(a,b);
}

template<>
inline __device__ float2 fminf<float2>(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}

template<>
inline __device__ float3 fminf<float3>(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

template<>
inline  __device__ float4 fminf<float4>(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

template<typename VectorType>
inline __host__ __device__ VectorType make_float(float s);

template<>
inline __host__ __device__ float make_float<float>(float s)
{
    return s;
}

template<>
inline __host__ __device__ float2 make_float<float2>(float s)
{
    return make_float2(s, s);
}

template<>
inline __host__ __device__ float3 make_float<float3>(float s)
{
    return make_float3(s, s, s);
}

template<>
inline __host__ __device__ float4 make_float<float4>(float s)
{
    return make_float4(s, s, s, s);
}

template<typename TexelType, typename VectorType>
__device__ __inline__ TexelType toType(VectorType vec);

template<>
inline __host__ __device__ ushort toType<ushort, float>(float v)
{
    return (ushort)v;
}

template<>
inline __host__ __device__ float toType<float, ushort>(ushort v)
{
    return (float)v;
}

template<>
inline __host__ __device__ ushort2 toType<ushort2, float2>(float2 vec)
{
    return make_ushort2((unsigned short)vec.x, (unsigned short)vec.y);
}

template<>
inline __host__ __device__ float2 toType<float2, ushort2>(ushort2 vec)
{
    return make_float2((float)vec.x, (float)vec.y);
}

template<>
inline __host__ __device__ ushort4 toType<ushort4, float4>(float4 vec)
{
    return make_ushort4((unsigned short)vec.x, (unsigned short)vec.y, (unsigned short)vec.z, (unsigned short)vec.w);
}

template<>
inline __host__ __device__ float4 toType<float4, ushort4>(ushort4 vec)
{
    return make_float4((float)vec.x, (float)vec.y, (float)vec.z, (float)vec.w);
}

template<>
inline __host__ __device__ uchar2 toType<uchar2, float2>(float2 vec)
{
    return make_uchar2((unsigned char)vec.x, (unsigned char)vec.y);
}

template<>
inline __host__ __device__ uchar4 toType<uchar4, float4>(float4 vec)
{
    return make_uchar4((unsigned char)vec.x, (unsigned char)vec.y, (unsigned char)vec.z, (unsigned char)vec.w);
}

template<typename TexelType>
__device__ __inline__ float GetMaxValue();

template<>
__device__ __inline__ float GetMaxValue<ushort4>() { return 65535.0f; };

template<>
__device__ __inline__ float GetMaxValue<ushort>() { return 65535.0f; };


template<typename VectorType, typename TexelType>
__global__ void MipmapGen(SurfObj mipOutput, SurfObj mipInput, uint imageW, uint imageH)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < imageW) && (y < imageH))
    {
        VectorType val0 = toType<VectorType, TexelType>(surf2Dread<TexelType>(mipInput, 2 * x * sizeof(TexelType), 2 * y, cudaBoundaryModeClamp));
        VectorType val1 = toType<VectorType, TexelType>(surf2Dread<TexelType>(mipInput, (2 * x + 1) * sizeof(TexelType), 2 * y, cudaBoundaryModeClamp));
        VectorType val2 = toType<VectorType, TexelType>(surf2Dread<TexelType>(mipInput, 2 * x * sizeof(TexelType), (2 * y + 1), cudaBoundaryModeClamp));
        VectorType val3 = toType<VectorType, TexelType>(surf2Dread<TexelType>(mipInput, (2 * x + 1) * sizeof(TexelType), (2 * y + 1), cudaBoundaryModeClamp));

        VectorType color = (val0 + val1 + val2 + val3) / 4.0f;

        color = fminf<VectorType>(color, make_float<VectorType>(GetMaxValue<TexelType>()));

        surf2Dwrite(toType<TexelType, VectorType>(color), mipOutput, x * sizeof(TexelType), y);
    }
}

inline uint getMipMapLevels(cudaExtent size)
{
    size_t sz = max(max(size.width,size.height),size.depth);

    uint levels = 0;

    while (sz)
    {
        sz /= 2;
        levels++;
    }

    return levels;
}

template<typename VectorType, typename TexelType>
void MipmapTexture::GenerateMipmap()
{
    uint level = 1;
    uint width = mipSizes[0].x;
    uint height = mipSizes[0].y;

    while ((width != 1 || height != 1) && level < numMipLevels)
    {
        width = max(1u, width / 2);
        height = max(1u, height / 2);
        assert(width == mipSizes[level].x);
        assert(height == mipSizes[level].y);

        dim3 blockSize(16, 16, 1);
		dim3 gridSize(divRoundUp(width, blockSize.x), divRoundUp(height, blockSize.y), 1);
		MipmapGen <VectorType, TexelType> <<<gridSize, blockSize >>> (mip[level], mip[level - 1], width, height);

        checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

        level++;
    }
}

template void MipmapTexture::GenerateMipmap<float, ushort>();
template void MipmapTexture::GenerateMipmap<float4, ushort4>();