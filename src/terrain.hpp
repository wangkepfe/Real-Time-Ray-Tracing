#pragma once

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include "geometry.h"

class Terrain
{
public:
    Terrain() :
        mapSize        { 32 },
        heightBase     { 8 },
        noiseAmpLimit  { 4 },
        baseFrequency  { 4 },
        aabbEdgeLength { 0.01f },
		heightMap(mapSize, 0)
    {
        srand(25678u);
    }

    ~Terrain() {}

    void generateHeightMap()
    {
        std::vector<float> noise(mapSize, 0);
		std::vector<float> gradient(mapSize, 0);

		unsigned int sampleCountLevel = baseFrequency;
		float currentAmp = 1.0f;

		while (sampleCountLevel < mapSize)
        {
            // generate gradient
            for (unsigned int i = 0; i < sampleCountLevel; ++i)
            {
                gradient[i] = GetRandMinusOneToOne() * currentAmp;
            }

            // sample perlin noise
            for (unsigned int i = 0; i < mapSize; ++i)
            {
                noise[i] += samplePerlinNoise(i / (float)mapSize * sampleCountLevel, gradient);
            }

			sampleCountLevel *= 2;
			currentAmp /= 2;
        }

        // float to int
        for (unsigned int i = 0; i < mapSize; ++i)
        {
            heightMap[i] = heightBase + (int)(roundf(noise[i] * (float)noiseAmpLimit));
        }
    }

	AABB* generateAabbs(int& numAabbs)
    {
        const float offsetX = -(mapSize * 0.5f * aabbEdgeLength);
        //const float offsetY = -((heightBase + noiseAmpLimit) * 0.5f * aabbEdgeLength);
        const float offsetY = 0;

        for (unsigned int i = 0; i < mapSize; ++i)
        {
            // for (int j = 0; j <= heightMap[i]; ++j)
            // {
			// 	float posX = i * aabbEdgeLength + offsetX;
			// 	float posY = j * aabbEdgeLength + offsetY;

			// 	Float3 pos(posX, posY, 0);

			// 	AABB aabbTemp(pos, aabbEdgeLength);

            //     aabbs.push_back(aabbTemp);
            // }
            for (int j = heightMap[i]; j >= 0; --j)
            {
				float posX = i * aabbEdgeLength + offsetX;
				float posY = j * aabbEdgeLength + offsetY;
                float posZ = (j - heightMap[i]) * aabbEdgeLength;

				Float3 pos(posX, posY, posZ);

				AABB aabbTemp(pos, aabbEdgeLength);

                aabbs.push_back(aabbTemp);
            }
        }

        numAabbs = (int)aabbs.size();
        return aabbs.data();
    }

	float getHeightAt(float pos)
	{
		const float offsetX = mapSize * 0.5f * aabbEdgeLength;
        //const float offsetY = -((heightBase + noiseAmpLimit - 1) * 0.5f * aabbEdgeLength);
        const float offsetY = 0.5f * aabbEdgeLength;
		return heightMap[(unsigned int)((pos + offsetX) / aabbEdgeLength)] * aabbEdgeLength + offsetY;
	}

private:

    float GetRandNum() { return rand() / (float)RAND_MAX; }
    float GetRandMinusOneToOne() { return GetRandNum() * 2.0f - 1.0f; }

    float samplePerlinNoise(float x, std::vector<float> slopes)
    {
        float lo = floorf(x);
        float hi = lo + 1.0f;
        float d = x - lo;
        float slopeLo = slopes[(unsigned int)lo];
        float slopeHi = slopes[(unsigned int)hi];
        float posLo = slopeLo * d;
        float posHi = - slopeHi * (1.0f - d);
        float d2 = d * d;
        float d3 = d2 * d;
        float u = (6.0f * d2 - 15.0f * d + 10.0f) * d3;
        return posLo * (1.0f - u) + posHi * u;
    }

    const unsigned int mapSize;
    const int          heightBase;
    const unsigned int noiseAmpLimit;
    const unsigned int baseFrequency;
    const float        aabbEdgeLength;

    std::vector<AABB>  aabbs;
    std::vector<int>   heightMap;
};



