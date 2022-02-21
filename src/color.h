#pragma once

#include <cuda_runtime.h>
#include "linearMath.h"

inline __device__ Float3 XyzToRgbAces2065(Float3 xyzColor)
{
	// ACES 2065-1 D60
	constexpr Mat3 xyzToRgb(
		1.0498110175f , 0.0f        , -0.0000974845f,
		-0.4959030231f, 1.3733130458f, 0.0982400361f,
		0.0f          , 0.0f        , 0.9912520182f);

	Float3 rgbColor = xyzToRgb * xyzColor;

	return rgbColor;
}

inline __device__ Float3 XyzToRgbSrgb(Float3 xyzColor)
{
	// SRGB D65
	constexpr Mat3 xyzToRgb(
		3.2404542, -1.5371385, -0.4985314,
		-0.9692660,  1.8760108,  0.0415560,
		0.0556434, -0.2040259,  1.0572252);

	Float3 rgbColor = xyzToRgb * xyzColor;

	return rgbColor;
}

inline __device__ Float3 Aces2065ToSrgb(Float3 color)
{
	// SRGB D65
	constexpr Mat3 xyzToRgb1(
		3.2404542, -1.5371385, -0.4985314,
		-0.9692660,  1.8760108,  0.0415560,
		0.0556434, -0.2040259,  1.0572252);

    constexpr Mat3 xyzToRgb2(
		1.0498110175f , 0.0f        , -0.0000974845f,
		-0.4959030231f, 1.3733130458f, 0.0982400361f,
		0.0f          , 0.0f        , 0.9912520182f);

	Float3 rgbColor = xyzToRgb1 * (Inverse(xyzToRgb2) * color);

	return rgbColor;
}