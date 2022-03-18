#pragma once

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_math.h>

struct __align__(16) MipmapImage
{
    cudaExtent              size;
    uint                    highestLod;
    void                   *h_data;
    cudaResourceType        type;
    cudaMipmappedArray_t    mipmapArray;
    cudaTextureObject_t     textureObject;
};

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
inline __host__ __device__ ushort2 toType<ushort2, float2>(float2 vec)
{
    return make_ushort2((unsigned short)vec.x, (unsigned short)vec.y);
}

template<>
inline __host__ __device__ ushort4 toType<ushort4, float4>(float4 vec)
{
    return make_ushort4((unsigned short)vec.x, (unsigned short)vec.y, (unsigned short)vec.z, (unsigned short)vec.w);
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

template<typename VectorType, typename TexelType>
__global__ void d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0/float(imageW);
    float py = 1.0/float(imageH);

    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples
        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.

        VectorType color =
            (tex2D<VectorType>(mipInput,(x + 0) * px, (y + 0) * py)) +
            (tex2D<VectorType>(mipInput,(x + 1) * px, (y + 0) * py)) +
            (tex2D<VectorType>(mipInput,(x + 1) * px, (y + 1) * py)) +
            (tex2D<VectorType>(mipInput,(x + 0) * px, (y + 1) * py));

        color *= (0.25f * 255.0f);
        color = fminf<VectorType>(color, make_float<VectorType>(255.0f));

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
inline void InitMipmapImage(MipmapImage* pImage)
{
    MipmapImage &image = *pImage;
    image.type = cudaResourceTypeMipmappedArray;

    // how many mipmaps we need
    uint levels = getMipMapLevels(image.size);
    image.highestLod = max(1.0f, (float) levels-1);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<TexelType>();
    checkCudaErrors(cudaMallocMipmappedArray(&image.mipmapArray, &desc, image.size, levels));

    // upload level 0
    cudaArray_t level0;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, image.mipmapArray, 0));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr       = make_cudaPitchedPtr(image.h_data, image.size.width * sizeof(TexelType), image.size.width, image.size.height);
    copyParams.dstArray     = level0;
    copyParams.extent       = image.size;
    copyParams.extent.depth = 1;
    copyParams.kind         = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    // compute rest of mipmaps based on level 0
	{
		cudaMipmappedArray_t mipmapArray = image.mipmapArray;
		cudaExtent size = image.size;

		size_t width = size.width;
		size_t height = size.height;

		uint level = 0;

		while (width != 1 || height != 1)
		{
			width /= 2;
			width = max((size_t)1, width);
			height /= 2;
			height = max((size_t)1, height);

			cudaArray_t levelFrom;
			checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
			cudaArray_t levelTo;
			checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

			cudaExtent  levelToSize;
			checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
			assert(levelToSize.width == width);
			assert(levelToSize.height == height);
			assert(levelToSize.depth == 0);

			// generate texture object for reading
			cudaTextureObject_t         texInput;
			cudaResourceDesc            texRes;
			memset(&texRes, 0, sizeof(cudaResourceDesc));

			texRes.resType = cudaResourceTypeArray;
			texRes.res.array.array = levelFrom;

			cudaTextureDesc             texDescr;
			memset(&texDescr, 0, sizeof(cudaTextureDesc));

			texDescr.normalizedCoords = 1;
			texDescr.filterMode = cudaFilterModeLinear;

			texDescr.addressMode[0] = cudaAddressModeClamp;
			texDescr.addressMode[1] = cudaAddressModeClamp;
			texDescr.addressMode[2] = cudaAddressModeClamp;

			texDescr.readMode = cudaReadModeNormalizedFloat;

			checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

			// generate surface object for writing

			cudaSurfaceObject_t surfOutput;
			cudaResourceDesc    surfRes;
			memset(&surfRes, 0, sizeof(cudaResourceDesc));
			surfRes.resType = cudaResourceTypeArray;
			surfRes.res.array.array = levelTo;

			checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));

			// run mipmap kernel
			dim3 blockSize(16, 16, 1);
			dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

			d_mipmap <VectorType, TexelType> << <gridSize, blockSize >> > (surfOutput, texInput, (uint)width, (uint)height);

			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());

			checkCudaErrors(cudaDestroySurfaceObject(surfOutput));

			checkCudaErrors(cudaDestroyTextureObject(texInput));

			level++;
		}
	}

    // generate bindless texture object
    cudaResourceDesc            resDescr;
    memset(&resDescr,0,sizeof(cudaResourceDesc));

    resDescr.resType            = cudaResourceTypeMipmappedArray;
    resDescr.res.mipmap.mipmap  = image.mipmapArray;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = 1;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.mipmapFilterMode = cudaFilterModeLinear;

    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    texDescr.maxMipmapLevelClamp = float(levels - 1);

    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(cudaCreateTextureObject(&image.textureObject, &resDescr, &texDescr, NULL));
}
