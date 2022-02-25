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