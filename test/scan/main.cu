
#include "testCommon.h"
#include "scan.cuh"

int main()
{
    gpuDeviceInit(0);

    // float n = 2;
    //for (int i = 0; i < 8; ++i)
    {
        const unsigned int size = 128 * 2048;
        const unsigned int kernelSize = 128;
		std::cout << "kernelSize = " << kernelSize << "\n";
		// n *= 2;
        const unsigned int blockSize = size / kernelSize;
        std::cout << "blockSize = " << blockSize << "\n";
        const int postfix = 1;

        float* h_in = new float[size];
        float* h_out = new float[size];
        float* cpuOut = new float[size];

        RandomArray(h_in, size);

        float* in;
        float* out;
        float* tmp;

        GpuErrorCheck(cudaMalloc((void**)& in, size * sizeof(float)));
        GpuErrorCheck(cudaMalloc((void**)& out, size * sizeof(float)));
        GpuErrorCheck(cudaMalloc((void**)& tmp, blockSize * sizeof(float)));

        GpuErrorCheck(cudaMemcpy(in, h_in, size * sizeof(float), cudaMemcpyHostToDevice));

        GpuErrorCheck(cudaDeviceSynchronize());

        {
            ScopeTimer timer("GPU Scan");
            Scan(in, out, tmp, size, kernelSize, postfix);
            GpuErrorCheck(cudaDeviceSynchronize());
            GpuErrorCheck(cudaPeekAtLastError());
        }

        GpuErrorCheck(cudaMemcpy(h_out, out, size * sizeof(float), cudaMemcpyDeviceToHost));

        {
            ScopeTimer timer("CPU Scan");
            CpuScan(h_in, cpuOut, size, postfix);
        }

        std::cout << cpuOut[size - 1] << "\n";
        std::cout << h_out[size - 1] << "\n";

        std::cout << "result = " << ArrayAlmostEqual(cpuOut, h_out, size, 5) << "\n";
        //PrintArray(h_out, size, 128);
        //PrintArray(cpuOut, size, 128);

        cudaFree(in);
        cudaFree(out);
        cudaFree(tmp);

        delete[] h_in;
        delete[] h_out;
        delete[] cpuOut;
    }

    return 0;
}