#pragma once

#include "kernel.cuh"
#include <iostream>
#include <vector>

void LoadScene(const char* filePath, std::vector<Triangle>& h_triangles);
cudaArray* LoadTextureRgba8(const char* texPath, cudaTextureObject_t& texObj);
cudaArray* LoadTextureRgb8(const char* texPath, cudaTextureObject_t& texObj);