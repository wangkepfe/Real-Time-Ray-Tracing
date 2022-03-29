#pragma once

#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include "fp16Utils.cuh"
#include "texture.h"

#define uchar unsigned char

//----------------------------------------------------------------------------------------------
//
//                                   Utils
//
//----------------------------------------------------------------------------------------------


union UintFloatConverter { uint ui; float f; __device__ UintFloatConverter() : ui(0) {} };
struct Float3Ushort1 { Float3 xyz; ushort w; };
struct Half3Ushort1 { Half3 xyz; ushort w; };


//----------------------------------------------------------------------------------------------
//
//                                   Load/Store Functions
//
//----------------------------------------------------------------------------------------------


//------------------------------------------- 2d load store ---------------------------------------------------
// For cuda types like float4

template<typename T>
__forceinline__ __device__ T Load2D (
	SurfObj tex,
	Int2    uv,
    cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeClamp)
{
    return surf2Dread<T>(tex, uv.x * sizeof(T), uv.y, boundaryMode);
}

template<typename T>
__forceinline__ __device__ void Store2D (
    T       val,
	SurfObj tex,
	Int2    uv,
    cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeClamp)
{
    surf2Dwrite(val, tex, uv.x * sizeof(T), uv.y, boundaryMode);
}

//------------------------------------------- 2d Float4 ---------------------------------------------------
// byte    4     4     4     4
// data  float float float float

__forceinline__ __device__ void Store2DFloat4 (
    Float4  val,
	SurfObj tex,
	Int2    uv)
{
    surf2Dwrite(make_float4(val.x, val.y, val.z, val.w), tex, uv.x * 4 * sizeof(float), uv.y, cudaBoundaryModeClamp);
}

__forceinline__ __device__ Float4 Load2DFloat4(
	SurfObj tex,
	Int2    uv)
{
    float4 ret = surf2Dread<float4>(tex, uv.x * 4 * sizeof(float), uv.y, cudaBoundaryModeClamp);
	return Float4(ret.x, ret.y, ret.z, ret.w);
}

//------------------------------------------- 2d Float2 ---------------------------------------------------
// byte    4     4
// data  float float

__forceinline__ __device__ Float2 Load2DFloat2(SurfObj tex, Int2 uv)
{
    float2 ret = surf2Dread<float2>(tex, uv.x * 2 * sizeof(float), uv.y, cudaBoundaryModeClamp);
	return Float2(ret.x, ret.y);
}

__forceinline__ __device__ void Store2DFloat2(Float2 val, SurfObj tex, Int2 uv)
{
    surf2Dwrite(make_float2(val.x, val.y), tex, uv.x * 2 * sizeof(float), uv.y, cudaBoundaryModeClamp);
}


//------------------------------------------- 2d Half3Ushort1 ---------------------------------------------------
// byte    2     2     2     2
// data  half  half  half  ushort

template<typename ReturnType = Float3Ushort1>
__forceinline__ __device__ ReturnType Load2DHalf3Ushort1(
	SurfObj tex,
	Int2    uv)
{
	ushort4 ret = surf2Dread<ushort4>(tex, uv.x * 4 * sizeof(unsigned short), uv.y, cudaBoundaryModeClamp);

	ushort4ToHalf4Converter conv(ret);
	Half4 hf4 = conv.hf4;
    Float4 fl4 = half4ToFloat4(hf4);

	return { fl4.xyz, ret.w };
}

template<>
__forceinline__ __device__ Half3Ushort1 Load2DHalf3Ushort1<Half3Ushort1>(
	SurfObj tex,
	Int2    uv)
{
	ushort4 ret = surf2Dread<ushort4>(tex, uv.x * 4 * sizeof(unsigned short), uv.y, cudaBoundaryModeClamp);
	ushort4ToHalf3Converter conv(ret);
	return { conv.hf3, ret.w };
}

__forceinline__ __device__ void Store2DHalf3Ushort1(
    Float3Ushort1 val,
	SurfObj       tex,
	Int2          uv)
{
    Half4 hf4 = float4ToHalf4(Float4(val.xyz, 0));

    ushort4ToHalf4Converter conv(hf4);
    ushort4 us4 = conv.us4;

    us4.w = val.w;

    surf2Dwrite(us4, tex, uv.x * 4 * sizeof(unsigned short), uv.y, cudaBoundaryModeClamp);
}

//------------------------------------------- 2d Half4 ---------------------------------------------------
// byte    2     2     2     2
// data  half  half  half  half

template<typename ReturnType = Float4>
__forceinline__ __device__ ReturnType Load2DHalf4(
	SurfObj tex,
	Int2    uv)
{
	ushort4 ret = surf2Dread<ushort4>(tex, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp);

	ushort4ToHalf4Converter conv(ret);
	Half4 hf4 = conv.hf4;

	return half4ToFloat4(hf4);
}

template<>
__forceinline__ __device__ Half4 Load2DHalf4<Half4>(
	SurfObj tex,
	Int2    uv)
{
	ushort4 ret = surf2Dread<ushort4>(tex, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp);

	ushort4ToHalf4Converter conv(ret);
	Half4 hf4 = conv.hf4;

	return hf4;
}

template<>
__forceinline__ __device__ Half3 Load2DHalf4<Half3>(
	SurfObj tex,
	Int2    uv)
{
	ushort4 ret = surf2Dread<ushort4>(tex, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp);

	ushort4ToHalf3Converter conv(ret);
	Half3 hf3 = conv.hf3;

	return hf3;
}

__forceinline__ __device__ void Store2DHalf4(
    Float4  fl4,
	SurfObj tex,
	Int2    uv)
{
    Half4 hf4 = float4ToHalf4(fl4);

    ushort4ToHalf4Converter conv(hf4);
    ushort4 us4 = conv.us4;

    surf2Dwrite(us4, tex, uv.x * 4 * sizeof(unsigned short), uv.y, cudaBoundaryModeClamp);
}

//------------------------------------------- 2d Ushort4 ---------------------------------------------------
// byte    2       2       2       2
// data  ushort  ushort  ushort  ushort

__forceinline__ __device__ Float4 Load2DUshort4(
	SurfObj tex,
	Int2    uv)
{
	ushort4 ret = surf2Dread<ushort4>(tex, uv.x * sizeof(ushort4), uv.y, cudaBoundaryModeClamp);
	return { ret.x / (float)USHRT_MAX,
             ret.y / (float)USHRT_MAX,
             ret.z / (float)USHRT_MAX,
             ret.w / (float)USHRT_MAX };
}

__forceinline__ __device__ void Store2DUshort4(
    Float4  value,
	SurfObj tex,
	Int2    uv)
{
    Float4 val = clamp4f(value) * (float)USHRT_MAX;
	surf2Dwrite(make_ushort4((ushort)val.x, (ushort)val.y, (ushort)val.z, (ushort)val.w), tex, uv.x * 4 * sizeof(ushort), uv.y, cudaBoundaryModeClamp);
}

//------------------------------------------- 2d half2 ---------------------------------------------------
// byte    2     2
// data  half  half

__forceinline__ __device__ Float2 Load2DHalf2(SurfObj tex, Int2 uv)
{
    ushort2ToHalf2Converter conv(surf2Dread<ushort2>(tex, uv.x * 2 * sizeof(short), uv.y, cudaBoundaryModeClamp));
    float2 ret = __half22float2(conv.hf2);
	return Float2(ret.x, ret.y);
}

__forceinline__ __device__ void Store2DHalf2(Float2 val, SurfObj tex, Int2 uv)
{
    ushort2ToHalf2Converter conv(__float22half2_rn(make_float2(val.x, val.y)));
    surf2Dwrite(conv.us2, tex, uv.x * 2 * sizeof(short), uv.y, cudaBoundaryModeClamp);
}

//------------------------------------------- 2d half1 ---------------------------------------------------
// byte    2
// data  half

template<typename ReturnType = float>
__forceinline__ __device__ ReturnType Load2DHalf1(SurfObj tex, Int2 uv)
{
    ushort1ToHalf1Converter conv(surf2Dread<ushort1>(tex, uv.x * 1 * sizeof(short), uv.y, cudaBoundaryModeClamp));
    return __half2float(conv.hf1);
}

template<>
__forceinline__ __device__ half Load2DHalf1<half>(SurfObj tex, Int2 uv)
{
    ushort1ToHalf1Converter conv(surf2Dread<ushort1>(tex, uv.x * 1 * sizeof(short), uv.y, cudaBoundaryModeClamp));
    return conv.hf1;
}

__forceinline__ __device__ void Store2DHalf1(float val, SurfObj tex, Int2 uv)
{
    ushort1ToHalf1Converter conv(__float2half(val));
    surf2Dwrite(conv.us1, tex, uv.x * 1 * sizeof(short), uv.y, cudaBoundaryModeClamp);
}

//------------------------------------------- 2d Ushort ---------------------------------------------------
// byte    2
// data  ushort

__forceinline__ __device__ float Load2DUshort1(
	SurfObj tex,
	Int2    uv)
{
	ushort1 ret = surf2Dread<ushort1>(tex, uv.x * sizeof(ushort1), uv.y, cudaBoundaryModeClamp);
	return ret.x / (float)USHRT_MAX;
}

__forceinline__ __device__ void Store2DUshort1(
    float  value,
	SurfObj tex,
	Int2    uv)
{
    float val = clampf(value) * (float)USHRT_MAX;
	surf2Dwrite(make_ushort1((ushort)val), tex, uv.x * sizeof(ushort1), uv.y, cudaBoundaryModeClamp);
}


//------------------------------------------- 2d uchar1 ---------------------------------------------------
// byte    1
// data  uchar

__forceinline__ __device__ int Load2DUchar1(SurfObj tex, Int2 uv)
{
    return surf2Dread<uchar1>(tex, uv.x, uv.y, cudaBoundaryModeClamp).x;
}

__forceinline__ __device__ void Store2DUchar1(int val, SurfObj tex, Int2 uv)
{
    surf2Dwrite(make_uchar1(val), tex, uv.x, uv.y, cudaBoundaryModeClamp);
}

//----------------------------------------------------------------------------------------------
//
//                                   Boundary Functors
//
//----------------------------------------------------------------------------------------------

struct BoundaryFuncDefault
{
    __forceinline__ __device__ Int2 operator() (Int2 uv, Int2 size) { return uv; }
};

struct BoundaryFuncClamp
{
    __forceinline__ __device__ Int2 operator() (Int2 uvIn, Int2 size)
    {
        Int2 uv = uvIn;

        if (uv.x >= size.x) { uv.x = size.x - 1; }
        if (uv.y >= size.y) { uv.y = size.y - 1; }
        if (uv.x < 0) { uv.x = 0; }
        if (uv.y < 0) { uv.y = 0; }

        return uv;
    }
};

struct BoundaryFuncRepeat
{
    __forceinline__ __device__ Int2 operator() (Int2 uvIn, Int2 size)
    {
        Int2 uv = uvIn;

        if (uv.x >= size.x) { uv.x %= size.x; }
        if (uv.y >= size.y) { uv.y %= size.y; }
        if (uv.x < 0) { uv.x = size.x - (-uv.x) % size.x; }
        if (uv.y < 0) { uv.y = size.y - (-uv.y) % size.y; }

        return uv;
    }
};

struct BoundaryFuncRepeatXClampY
{
    __forceinline__ __device__ Int2 operator() (Int2 uvIn, Int2 size)
    {
        Int2 uv = uvIn;

        if (uv.x >= size.x) { uv.x %= size.x; }
        if (uv.x < 0) { uv.x = size.x - (-uv.x) % size.x; }
        if (uv.y >= size.y) { uv.y = size.y - 1; }
        if (uv.y < 0) { uv.y = 0; }

        return uv;
    }
};

//----------------------------------------------------------------------------------------------
//
//                                   Load Functors
//
//----------------------------------------------------------------------------------------------


template<typename VectorType = Float3>
struct Load2DFuncHalf3Ushort1
{
    __forceinline__ __device__ VectorType operator()(SurfObj tex, Int2 uv) { return Load2DHalf3Ushort1(tex, uv).xyz; }
};

template<typename VectorType = Float3>
struct Load2DFuncHalf4
{
    __forceinline__ __device__ VectorType operator()(SurfObj tex, Int2 uv) { return Load2DHalf4(tex, uv).xyz; }
};

template<typename VectorType = Float3>
struct Load2DFuncFloat4
{
    __forceinline__ __device__ VectorType operator()(SurfObj tex, Int2 uv) { return Load2DFloat4(tex, uv).xyz; }
};

template<typename VectorType = Float3>
struct Load2DFuncUshort4
{
    __forceinline__ __device__ VectorType operator()(SurfObj tex, Int2 uv) { return Load2DUshort4(tex, uv).xyz; }
};

template<>
struct Load2DFuncUshort4<Float4>
{
    __forceinline__ __device__ Float4 operator()(SurfObj tex, Int2 uv) { return Load2DUshort4(tex, uv); }
};

template<typename VectorType = float>
struct Load2DFuncUshort1
{
    __forceinline__ __device__ VectorType operator()(SurfObj tex, Int2 uv) { return Load2DUshort1(tex, uv); }
};

//----------------------------------------------------------------------------------------------
//
//                                   Sample Functions
//
//----------------------------------------------------------------------------------------------

template<typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
inline __device__ VectorType SampleNearest(
    SurfObj             tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV  = uv * texSize;
    Int2 tc    = floori(UV + 0.5f);
    return LoadFunc()(tex, BoundaryFunc()(tc, texSize));
}

template<typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
inline __device__ VectorType SampleBilinear(
    SurfObj             tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV         = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc         = floor( UV - 0.5f ) + 0.5f;
    Float2 f          = UV - tc;

    Float2 w1 = f;
    Float2 w0 = 1.0f - f;

    Int2 tc0 = floori(UV - 0.5f);
    Int2 tc1 = tc0 + 1;

    Int2 sampleUV[4] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y },
    };

    float weights[4] = {
        w0.x * w0.y,  w1.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,
    };

	VectorType OutColor;
    float sumWeight = 0;

	#pragma unroll
	for (int i = 0; i < 4; i++)
	{
        sumWeight += weights[i];
		OutColor += LoadFunc()(tex, BoundaryFunc()(sampleUV[i], texSize)) * weights[i];
	}

	OutColor /= sumWeight;

    return OutColor;
}

template<typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
inline __device__ VectorType SampleBicubicCatmullRom(
    SurfObj             tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV         = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc         = floor( UV - 0.5f ) + 0.5f;
    Float2 f          = UV - tc;

	Float2 f2 = f * f;
	Float2 f3 = f2 * f;

    Float2 w0 = f2 - 0.5f * (f3 + f);
	Float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
	Float2 w3 = 0.5f * (f3 - f2);
	Float2 w2 = 1.0f - w0 - w1 - w3;

    Int2 tc1 = floori(UV - 0.5f);
    Int2 tc0 = tc1 - 1;
    Int2 tc2 = tc1 + 1;
    Int2 tc3 = tc1 + 2;

    Int2 sampleUV[16] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y }, { tc2.x, tc0.y }, { tc3.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y }, { tc2.x, tc1.y }, { tc3.x, tc1.y },
        { tc0.x, tc2.y }, { tc1.x, tc2.y }, { tc2.x, tc2.y }, { tc3.x, tc2.y },
        { tc0.x, tc3.y }, { tc1.x, tc3.y }, { tc2.x, tc3.y }, { tc3.x, tc3.y },
    };

    float weights[16] = {
        w0.x * w0.y,  w1.x * w0.y,  w2.x * w0.y,  w3.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,  w2.x * w1.y,  w3.x * w1.y,
        w0.x * w2.y,  w1.x * w2.y,  w2.x * w2.y,  w3.x * w2.y,
        w0.x * w3.y,  w1.x * w3.y,  w2.x * w3.y,  w3.x * w3.y,
    };

	VectorType OutColor;
    float sumWeight = 0;

	#pragma unroll
	for (int i = 0; i < 16; i++)
	{
        sumWeight += weights[i];
		OutColor += LoadFunc()(tex, BoundaryFunc()(sampleUV[i], texSize)) * weights[i];
	}

	OutColor /= sumWeight;

    return OutColor;
}

template<typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
inline __device__ VectorType SampleBicubicSmoothStep(
    SurfObj             tex,
    const Float2&       uv,
    const Int2&         texSize)
{
    Float2 UV         = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc         = floor( UV - 0.5f ) + 0.5f;
    Float2 f          = UV - tc;

	Float2 f2 = f * f;
	Float2 f3 = f2 * f;

    Float2 w1 = -2.0f * f3 + 3.0f * f2;
    Float2 w0 = 1.0f - w1;

    Int2 tc0 = floori(UV - 0.5f);
    Int2 tc1 = tc0 + 1;

    Int2 sampleUV[4] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y },
    };

    float weights[4] = {
        w0.x * w0.y,  w1.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,
    };

	VectorType OutColor;
    float sumWeight = 0;

	#pragma unroll
	for (int i = 0; i < 4; i++)
	{
        sumWeight += weights[i];
		OutColor += LoadFunc()(tex, BoundaryFunc()(sampleUV[i], texSize)) * weights[i];
	}

	OutColor /= sumWeight;

    return OutColor;
}


template<typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
inline __device__ VectorType SampleNearestLod(
    Mipmap              tex,
    const Float2&       uv,
    float               lod)
{
    float low = floorf(lod);
    float f = lod - low;

    int loLod = (int) low;
    int hiLod = loLod + 1;

    loLod = clampi(loLod, 0, tex.maxLod);
    hiLod = clampi(hiLod, 0, tex.maxLod);

    VectorType valueLo = SampleNearest<LoadFunc, VectorType, BoundaryFunc>(tex.mip[loLod], uv, tex.size[loLod]);
    VectorType valueHi = SampleNearest<LoadFunc, VectorType, BoundaryFunc>(tex.mip[hiLod], uv, tex.size[hiLod]);

    return valueLo * (1.0f - f) + valueHi * f;
}

template<typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
inline __device__ VectorType SampleBicubicSmoothStepLod(
    Mipmap              tex,
    const Float2&       uv,
    float               lod)
{
    float low = floorf(lod);
    float f = lod - low;

    int loLod = (int) low;
    int hiLod = loLod + 1;

    loLod = clampi(loLod, 0, tex.maxLod);
    hiLod = clampi(hiLod, 0, tex.maxLod);

    VectorType valueLo = SampleBicubicSmoothStep<LoadFunc, VectorType, BoundaryFunc>(tex.mip[loLod], uv, tex.size[loLod]);
    VectorType valueHi = SampleBicubicSmoothStep<LoadFunc, VectorType, BoundaryFunc>(tex.mip[hiLod], uv, tex.size[hiLod]);

    return valueLo * (1.0f - f) + valueHi * f;
}


//----------------------------------------------------------------------------------------------
//
//                                   Encode/Decode R11_G10_B11
//
//----------------------------------------------------------------------------------------------

// [x sign] [x] [y sign] [y] [z sign] [z]
//    1     10     1      9     1     10

__device__ __inline__ float EncodeNormal_R11_G10_B11(Float3 normal)
{
    const uint max9 = (0x1 << 9) - 1;
    const uint max10 = (0x1 << 10) - 1;

    UintFloatConverter converter;
    converter.ui =
		((normal.x < 0) ? (0x1 << 31) : 0) | ((uint)(fabsf(normal.x) * max10) << 21) |
		((normal.y < 0) ? (0x1 << 20) : 0) | ((uint)(fabsf(normal.y) * max9) << 11)  |
		((normal.z < 0) ? (0x1 << 10) : 0) | ((uint)(fabsf(normal.z) * max10));
    return converter.f;
}

__device__ __inline__ Float3 DecodeNormal_R11_G10_B11(float fcode)
{
	const uint max9 = (0x1 << 9) - 1;
	const uint max10 = (0x1 << 10) - 1;

    const uint maskX = 0x7fe00000; // 0111 1111 1110 0000 0000 0000 0000 0000
    const uint maskY = 0x000ff800; // 0000 0000 0000 1111 1111 1000 0000 0000
    const uint maskZ = 0x000003ff; // 0000 0000 0000 0000 0000 0011 1111 1111

    UintFloatConverter converter;
    converter.f = fcode;
	uint code = converter.ui;

    return Float3(
		((code & (0x1 << 31) != 0) ? (-1) : 1) * (float)((code & maskX) >> 21) / (float)max10,
		((code & (0x1 << 20) != 0) ? (-1) : 1) * (float)((code & maskY) >> 11) / (float)max9,
		((code & (0x1 << 10) != 0) ? (-1) : 1) * (float)((code & maskZ)) / (float)max10);
}