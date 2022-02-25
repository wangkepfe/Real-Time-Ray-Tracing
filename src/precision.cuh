#pragma once

#include <cuda_runtime.h>

union flt_32 {
	float f32;
	int i32;
};

union int_32 {
	int i32;
	float f32;
};

// ------------------------------ Machine Epsilon -----------------------------------------------
// The smallest number that is larger than one minus one. ULP (unit in the last place) of 1
// ----------------------------------------------------------------------------------------------
__host__ __device__ __inline__ constexpr float MachineEpsilon()
{
	flt_32 s{ 1.0f };
	s.i32++;
	return (s.f32 - 1.0f);
}

// ------------------------------ Error Gamma -------------------------------------------------------
// return 32bit floating point arithmatic calculation error upper bound, n is number of calculation
// --------------------------------------------------------------------------------------------------
__host__ __device__ __inline__ constexpr float ErrGamma(int n)
{
	return (n * MachineEpsilon()) / (1.0f - n * MachineEpsilon());
}


__device__ __host__ inline const float TwoPowerMinus23()
{
	int_32 v {0x34000000};
	return v.f32;
}

__device__ __host__ inline const float TwoPowerMinus24()
{
	int_32 v {0x33800000};
	return v.f32;
}

__device__ __host__ inline const float p()
{
	return 1.0f + TwoPowerMinus23();
}

__device__ __host__ inline const float m()
{
	return 1.0f - TwoPowerMinus23();
}

__device__ __host__ inline float up(float a)
{
	return a>0.0f ? a*p() : a*m();
}

__device__ __host__ inline float dn(float a)
{
	return a>0.0f ? a*m() : a*p();
}

__device__ __host__ inline float Up(float a) { return a*p(); }
__device__ __host__ inline float Dn(float a) { return a*m(); }

__device__ __host__ inline Float3 Up(Float3 a) { return Float3(Up(a.x), Up(a.y), Up(a.z)); }
__device__ __host__ inline Float3 Dn(Float3 a) { return Float3(Dn(a.x), Dn(a.y), Dn(a.z)); }
