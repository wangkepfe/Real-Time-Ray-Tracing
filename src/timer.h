#pragma once

#include <chrono>

struct Timer
{
    Timer() {
        init();
    }

    void init() {
        previousTime = std::chrono::high_resolution_clock::now();
    }

    void update() {
        ++frameCounter;

        currentTime = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<double, std::milli>(currentTime - previousTime).count();
        previousTime = currentTime;

        fpsTimer += static_cast<float>(deltaTime);

        if (fpsTimer > 1000.0f) {
            fps = static_cast<uint32_t>(static_cast<float>(frameCounter) * (1000.0f / fpsTimer));
            fpsTimer -= 1000.0f;
            frameCounter = 0;
        }
    }

    float getDeltaTime() {
        return static_cast<float>(deltaTime);
    }

    float getTime() {
        return std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    }

    bool firstTimeUse = true;
    double deltaTime = 0.0;
    float fpsTimer = 0.0f;
    uint32_t frameCounter = 0;
    uint32_t fps = 0;

    const std::chrono::steady_clock::time_point startTime {std::chrono::high_resolution_clock::now()};
    std::chrono::steady_clock::time_point currentTime;
    std::chrono::steady_clock::time_point previousTime;
};