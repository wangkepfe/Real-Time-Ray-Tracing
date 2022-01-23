#pragma once

#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "helper_cuda.h"
#include "cudaError.cuh"
#include "timer.h"
#include "linear_math.h"

void RandomArray(float* a, int size)
{
    srand(time(0));

	for (int i = 0; i < size; ++i)
	{
		int r = rand();
		a[i] = r / (float)RAND_MAX;
	}
        
}

bool ArrayEqual(float* a, float* b, int size)
{
	for (int i = 0; i < size; ++i)
	{
		if (a[i] != b[i])
		{
			return false;
		}
	}
	return true;
}


bool ArrayAlmostEqual(float* a, float* b, int size, float errorPercentage)
{
	for (int i = 0; i < size; ++i)
	{
		float c, d;
		if (a[i] > b[i])
		{
			c = a[i];
			d = b[i];
		}
		else
		{
			c = b[i];
			d = a[i];
		}

		if (c / d - 1.0f > errorPercentage / 100.0f)
		{
			return false;
		}
	}
	return true;
}

void PrintArray(float* a, int size, int skip)
{
    for (int i = 0; i < size; i += skip)
	{
        std::cout << a[i] << " ";
    }
    std::cout << "\n\n";
}