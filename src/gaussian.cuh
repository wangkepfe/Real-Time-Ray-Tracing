#pragma once

#define GAUSSIAN_3x3_SIGMA 1.0f
#define GAUSSIAN_5x5_SIGMA 1.0f
#define GAUSSIAN_7x7_SIGMA 1.0f

#define PRINT_GAUSSIAN_KERNAL 1
#define USE_PRECALCULATED_GAUSSIAN 1

#if USE_PRECALCULATED_GAUSSIAN

inline __device__ float GetGaussian3x3(uint i) {
	float cGaussian3x3[] = {
		0.0578968, 0.0921378, 0.0584323,
		0.0921378, 0.146629, 0.09299,
		0.0584322, 0.0929898, 0.0589727
	};
	return cGaussian3x3[i];
}

inline __device__ float GetGaussian5x5(uint i) {
	float cGaussian5x5[] = {
		0.00360466, 0.0144464, 0.0229902, 0.01458, 0.0036719,
		0.0144464, 0.0578968, 0.0921378, 0.0584323, 0.0147159,
		0.0229902, 0.0921378, 0.146629, 0.09299, 0.023419,
		0.01458, 0.0584322, 0.0929898, 0.0589727, 0.014852,
		0.00367191, 0.0147158, 0.0234191, 0.0148519, 0.0037404
	};
	return cGaussian5x5[i];
}

inline __device__ float GetGaussian7x7(uint i) {
	float cGaussian7x7[] = {
		3.47404e-05, 0.000353875, 0.00141822, 0.00225698, 0.00143134, 0.000360475, 3.57221e-05,
		0.000353875, 0.00360466, 0.0144464, 0.0229902, 0.01458, 0.0036719, 0.000363875,
		0.00141822, 0.0144464, 0.0578968, 0.0921378, 0.0584323, 0.0147159, 0.0014583,
		0.00225698, 0.0229902, 0.0921378, 0.146629, 0.09299, 0.023419, 0.00232076,
		0.00143134, 0.01458, 0.0584322, 0.0929898, 0.0589727, 0.014852, 0.00147179,
		0.000360475, 0.00367191, 0.0147158, 0.0234191, 0.0148519, 0.0037404, 0.000370662,
		3.57221e-05, 0.000363875, 0.0014583, 0.00232075, 0.00147179, 0.000370662, 3.67315e-05
	};
	return cGaussian7x7[i];
}

#else

__constant__ float cGaussian3x3[9];  // 9
__constant__ float cGaussian5x5[25]; // 25
__constant__ float cGaussian7x7[49]; // 49

inline __device__ float GetGaussian3x3(uint i) {
	return cGaussian3x3[i];
}

inline __device__ float GetGaussian5x5(uint i) {
	return cGaussian5x5[i];
}

inline __device__ float GetGaussian7x7(uint i) {
	return cGaussian7x7[i];
}

inline float GaussianIsotropic2D(float x, float y, float sigma)
{
	return expf(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
}

inline void CalculateGaussianKernel(float* fGaussian, float sigma, int radius)
{
    int size = radius * 2 + 1;
    const int step = 100;
    int kernelSize = size * size;
    int sampleDimSize = size * step;
    int sampleCount = sampleDimSize * sampleDimSize;
    float *sampleData = new float[sampleCount];
    for (int i = 0; i < sampleCount; ++i)
    {
        int xi = i % sampleDimSize;
        int yi = i / sampleDimSize;
        float x = (float)xi / (float)step;
        float y = (float)yi / (float)step;
        float offset = (float)size / 2;
        x -= offset;
        y -= offset;
        sampleData[i] = GaussianIsotropic2D(x, y, sigma);
    }
    for (int i = 0; i < kernelSize; ++i)
    {
        int xi = i % size;
        int yi = i / size;
        float valSum = 0;
        for (int x = xi * step; x < (xi + 1) * step; ++x)
        {
            for (int y = yi * step; y < (yi + 1) * step; ++y)
            {
                valSum += sampleData[y * sampleDimSize + x];
            }
        }
        fGaussian[i] = valSum / (step * step);
    }
	if (PRINT_GAUSSIAN_KERNAL)
	{
		std::cout << "Gaussian " << size << "x" << size << "\n";
		std::cout << "sigma = " << sigma << "\n";
		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < size; ++j)
			{
				std::cout << fGaussian[i + j * size] << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
    delete sampleData;
}

inline void CalculateGaussian3x3()
{
	int kernelSize = 9;
	float* fGaussian = new float[kernelSize];
	CalculateGaussianKernel(fGaussian, GAUSSIAN_3x3_SIGMA, 1);
	GpuErrorCheck(cudaMemcpyToSymbol(cGaussian3x3, fGaussian, sizeof(float) * kernelSize));
	delete fGaussian;
}

inline void CalculateGaussian5x5()
{
	int kernelSize = 25;
	float* fGaussian = new float[kernelSize];
	CalculateGaussianKernel(fGaussian, GAUSSIAN_5x5_SIGMA, 2);
	GpuErrorCheck(cudaMemcpyToSymbol(cGaussian5x5, fGaussian, sizeof(float) * kernelSize));
	delete fGaussian;
}

inline void CalculateGaussian7x7()
{
	int kernelSize = 49;
	float* fGaussian = new float[kernelSize];
	CalculateGaussianKernel(fGaussian, GAUSSIAN_7x7_SIGMA, 3);
	GpuErrorCheck(cudaMemcpyToSymbol(cGaussian7x7, fGaussian, sizeof(float) * kernelSize));
	delete fGaussian;
}

#endif