#pragma once
#include <iostream>
#include <vector>

struct GlobalSettings
{
	int                      width;
	int                      height;

	std::string              inputMeshFileName;
    std::vector<std::string> inputTextureFileNames;
    bool                     loadCameraAtInit;
    std::string              inputCameraFileName;
    std::string              cameraSaveFileName;

    bool                     useDynamicResolution;
    float                    targetFps;
    int                      maxWidth;
    int                      maxHeight;
    int                      minWidth;
    int                      minHeight;
};
