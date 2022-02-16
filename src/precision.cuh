#pragma once

#include <cuda_runtime.h>

// ------------------------------ Machine Epsilon -----------------------------------------------
// The smallest number that is larger than one minus one. ULP (unit in the last place) of 1
// ----------------------------------------------------------------------------------------------
__host__ __device__ __inline__ constexpr float MachineEpsilon()
{
	typedef union {
		float f32;
		int i32;
	} flt_32;

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