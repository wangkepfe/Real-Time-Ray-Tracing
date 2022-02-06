#pragma once

#include <cuda_runtime.h>
#include "geometry.cuh"
#include "debugUtil.h"
#include "sampler.cuh"
#include "water.cuh"

// Return random noise in the range [0.0, 1.0], as a function of x.
inline __device__ float StarNoise2d(const Float2& v)
{
    float xhash = cosf( v.x * 37.0 );
    float yhash = cosf( v.y * 57.0 );
	float intPart;
	float val = modff( 415.92653 * ( xhash + yhash ) , &intPart);
	val = val < 0 ? val + 1 : val;
    return val;
}

// Convert Noise2d() into a "star field" by stomping everthing below fThreshhold to zero.
inline __device__ float NoisyStarField(const Float2& vSamplePos, float fThreshhold )
{
    float StarVal = StarNoise2d( vSamplePos );

    if ( StarVal >= fThreshhold )
        StarVal = powf( (StarVal - fThreshhold) / (1.0 - fThreshhold), 6.0 );
    else
        StarVal = 0.0;
    return StarVal;
}

// Stabilize NoisyStarField() by only sampling at integer values.
inline __device__ float StableStarField(const Float2& vSamplePos, float fThreshhold )
{
    // Linear interpolation between four samples.
    // Note: This approach has some visual artifacts.
    // There must be a better way to "anti alias" the star field.
	float intPart;

    float fractX = modff( vSamplePos.x , &intPart);
    float fractY = modff( vSamplePos.y , &intPart);

	fractX = fractX < 0 ? fractX + 1 : fractX;
	fractY = fractY < 0 ? fractY + 1 : fractY;

    Float2 floorSample = floor( vSamplePos );
    float v1 = NoisyStarField( floorSample, fThreshhold );
    float v2 = NoisyStarField( floorSample + Float2( 0.0, 1.0 ), fThreshhold );
    float v3 = NoisyStarField( floorSample + Float2( 1.0, 0.0 ), fThreshhold );
    float v4 = NoisyStarField( floorSample + Float2( 1.0, 1.0 ), fThreshhold );

    float StarVal =   v1 * ( 1.0 - fractX ) * ( 1.0 - fractY )
        			+ v2 * ( 1.0 - fractX ) * fractY
        			+ v3 * fractX * ( 1.0 - fractY )
        			+ v4 * fractX * fractY;
	return StarVal;
}
