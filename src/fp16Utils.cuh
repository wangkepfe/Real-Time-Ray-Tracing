#pragma once

#include "cuda_fp16.h"
#include "linear_math.h"

struct Half4
{
    half2 a;
    half2 b;
};

inline __device__ Float4 half4ToFloat4(Half4 v) {
    float2 a = __half22float2(v.a);
    float2 b = __half22float2(v.b);

    Float4 out;
    out.x = a.x;
    out.y = a.y;
    out.z = b.x;
    out.w = b.y;

    return out;
}

inline __device__ Half4 float4ToHalf4(Float4 v)
{
    float2 a;
    a.x = v.x;
    a.y = v.y;

    float2 b;
    b.x = v.z;
    b.y = v.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);

    return out;
}

union ushort4ToHalf4Converter
{
	__device__ ushort4ToHalf4Converter(const ushort4& v) : us4{v} {}
	__device__ ushort4ToHalf4Converter(const Half4& v) : hf4{v} {}

	ushort4 us4;
	Half4 hf4;
};

union ushort2ToHalf2Converter
{
    __device__ ushort2ToHalf2Converter(const ushort2& v) : us2{v} {}
	__device__ ushort2ToHalf2Converter(const half2& v) : hf2{v} {}

	ushort2 us2;
	half2 hf2;
};

union ushort1ToHalf1Converter
{
    __device__ ushort1ToHalf1Converter(const ushort1& v) : us1{v} {}
	__device__ ushort1ToHalf1Converter(const half& v) : hf1{v} {}

	ushort1 us1;
	half hf1;
};