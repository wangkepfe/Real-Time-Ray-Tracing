#pragma once

#include "kernel.cuh"
#include <iostream>
#include <vector>
#include <memory>

void LoadScene(const char* filePath, std::vector<Triangle>& h_triangles);

// struct LoadTexture8
// {
//     uint8_t* operator()(const char* texPath, int& texWidth, int& texHeight, int& texChannel, int nChannel);
// };

// struct LoadTexture16
// {
//     uint16_t* operator()(const char* texPath, int& texWidth, int& texHeight, int& texChannel, int nChannel);
// };