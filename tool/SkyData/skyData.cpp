#include <iostream>
#include <fstream>

#include "../../ref/skymodel/ArHosekSkyModelData_Spectral.h"
#include "../../ref/spectrum.h"

using namespace std;

int main ()
{
    ofstream myfile;
    myfile.open ("skyData.h");

    int turbidity = 4;
    int albedo = 1;
	myfile << "#pragma once\n";
    myfile << "static const float skyDataSets[] = {\n";
    for (double* dataset : datasets)
    {
        if (dataset == datasets[0]) continue;

        int beginIdx = albedo * 6 * 9 * 10 + (turbidity - 1) * 6 * 9;
        int endIdx = albedo * 6 * 9 * 10 + (turbidity) * 6 * 9;

        for (int i = beginIdx; i < endIdx; i += 9)
        {
            myfile << "    ";
            for (int j = i; j < i + 9; ++j)
            {
                myfile << std::scientific << static_cast<float>(dataset[j]) << "f, ";
            }
            myfile << endl;
        }
        myfile << endl;
    }
    myfile << "};\n\n";


    myfile << "static const float skyDataSetsRad[] = {\n";
    for (double* dataset : datasetsRad)
    {
        if (dataset == datasetsRad[0]) continue;

        int beginIdx = albedo * 6 * 10 + (turbidity - 1) * 6;
        int endIdx = albedo * 6 * 10 + (turbidity) * 6;

        for (int i = beginIdx; i < endIdx; i += 6)
        {
            myfile << "    ";
            for (int j = i; j < i + 6; ++j)
            {
                myfile << std::scientific << static_cast<float>(dataset[j]) << "f, ";
            }
            myfile << endl;
        }
        myfile << endl;
    }
    myfile << "};\n\n";


    myfile << "static const float h_solarDatasets[] = {\n";
    for (double* dataset : solarDatasets)
    {
        if (dataset == solarDatasets[0]) continue;

        int beginIdx = (turbidity - 1) * 4 * 45;
        int endIdx = (turbidity) * 4 * 45;

        for (int i = beginIdx; i < endIdx; i += 45)
        {
            myfile << "    ";
            for (int j = i; j < i + 45; ++j)
            {
                myfile << std::scientific << static_cast<float>(dataset[j]) << "f, ";
            }
            myfile << endl;
        }
        myfile << endl;
    }
    myfile << "};\n\n";


    myfile << "static const float h_limbDarkeningDatasets[] = {\n";
    for (double* dataset : limbDarkeningDatasets)
    {
        if (dataset == limbDarkeningDatasets[0]) continue;

        myfile << "    ";
        for (int j = 0; j < 6; ++j)
        {
            myfile << std::scientific << static_cast<float>(dataset[j]) << "f, ";
        }
        myfile << endl;
    }
    myfile << "};\n\n";


    int wl[10] = { 360, 400, 440, 480, 520, 560, 600, 640, 680, 720 };
    float valueX[10];
    float valueY[10];
    float valueZ[10];

    auto getSpectrumValue = [](const float* data, int wl) {
        if (wl < 360 || wl > 830) return 0.0f;
        return data[wl - 360];
    };

    int delta = wl[1] - wl[0];

    for (int i = 0; i < 10; ++i)
    {
        valueX[i] = 0;
        valueY[i] = 0;
        valueZ[i] = 0;

        int w = wl[i];

        for (int j = w - delta; j < w + delta; ++j)
        {
            float linearInterpFactor = 1.0f - (abs(w - j) / (float)delta);

            float v = getSpectrumValue(CIE_X, j);
            valueX[i] += v * linearInterpFactor;

            v = getSpectrumValue(CIE_Y, j);
            valueY[i] += v * linearInterpFactor;

            v = getSpectrumValue(CIE_Z, j);
            valueZ[i] += v * linearInterpFactor;
        }
    }

    myfile << "const float spectrumCieX[] = {\n    ";
    for (int i = 0; i < 10; ++i)
    {
        myfile << std::scientific << static_cast<float>(valueX[i]) << "f, ";
    }
    myfile << "\n};\n\n";

    myfile << "const float spectrumCieY[] = {\n    ";
    for (int i = 0; i < 10; ++i)
    {
        myfile << std::scientific << static_cast<float>(valueY[i]) << "f, ";
    }
    myfile << "\n};\n\n";

    myfile << "const float spectrumCieZ[] = {\n    ";
    for (int i = 0; i < 10; ++i)
    {
        myfile << std::scientific << static_cast<float>(valueZ[i]) << "f, ";
    }
    myfile << "\n};\n\n";

    myfile.close();

    return 0;
}