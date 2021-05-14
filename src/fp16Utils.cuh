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

union ushortToHalf
{
    __device__ ushortToHalf(const ushort& v) : us{v} {}
	__device__ ushortToHalf(const half& v) : hf{v} {}

    __device__ explicit operator ushort() { return us; }
    __device__ explicit operator half() { return hf; }

	ushort us;
	half hf;
};

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
//
//union ReconstructionInfo
//{
//	__device__ ReconstructionInfo() : all{make_uint4(0,0,0,0)} { _matId=0xffff; }
//    __device__ ReconstructionInfo(uint4 all) : all{all} {}
//
//	__device__ void packLi(Float3 Li)
//	{
//		Lix_Liy = __float22half2_rn(make_float2(Li.x, Li.y));
//		Liz = __float2half_rn(Li.z);
//	}
//
//	__device__ void packMatId(ushort matId)
//	{
//		_matId = matId;
//	}
//
//	__device__ void packCosines(float cosThetaWoWh, float cosThetaWo, float cosThetaWi, float cosThetaWh)
//	{
//		cosThetaWoWh_cosThetaWo = __float22half2_rn(make_float2(cosThetaWoWh, cosThetaWo));
//		cosThetaWi_cosThetaWh = __float22half2_rn(make_float2(cosThetaWi, cosThetaWh));
//	}
//
//	__device__ void unpackLi(Float3& Li)
//	{
//		float2 Lix_Liy_f = __half22float2(Lix_Liy);
//		float Liz_f = __half2float(Liz);
//		Li = Float3(Lix_Liy_f.x, Lix_Liy_f.y, Liz_f);
//	}
//
//	__device__ void unpackMatId(ushort& matId)
//	{
//		matId = _matId;
//	}
//
//	__device__ void unpackCosines(float& cosThetaWoWh, float& cosThetaWo, float& cosThetaWi, float& cosThetaWh)
//	{
//		float2 cosThetaWoWh_cosThetaWo_f = __half22float2(cosThetaWoWh_cosThetaWo);
//		float2 cosThetaWi_cosThetaWh_f = __half22float2(cosThetaWi_cosThetaWh);
//		cosThetaWoWh = cosThetaWoWh_cosThetaWo_f.x;
//		cosThetaWo = cosThetaWoWh_cosThetaWo_f.y;
//		cosThetaWi = cosThetaWi_cosThetaWh_f.x;
//		cosThetaWh = cosThetaWi_cosThetaWh_f.y;
//	}
//
//	struct __align__(16)
//	{
//		half2 Lix_Liy;
//		half Liz;
//		ushort _matId;
//
//		half2 cosThetaWoWh_cosThetaWo;
//		half2 cosThetaWi_cosThetaWh;
//
//		half2
//	};
//	uint4 all;
//};