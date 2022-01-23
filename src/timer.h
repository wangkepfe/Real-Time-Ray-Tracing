#pragma once

#include <chrono>
#include <ctime>
#include <iostream>

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

    static std::string getTimeString()
    {
        auto end = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::string str = std::ctime(&end_time);
        for (auto& c : str)
        {
            if (c == ' ') c = '-';
            if (c == ':') c = '-';
        }
        return str.substr(0, str.size() - 1);
    }

    bool     firstTimeUse = true;
    double   deltaTime    = 0.0;
    float    fpsTimer     = 0.0f;
    uint32_t frameCounter = 0;
    uint32_t fps          = 0;

    const std::chrono::steady_clock::time_point startTime {std::chrono::high_resolution_clock::now()};
    std::chrono::steady_clock::time_point currentTime;
    std::chrono::steady_clock::time_point previousTime;
};

struct ScopeTimer
{
    ScopeTimer(const std::string& name) : 
        name {name}, 
        startTime {std::chrono::high_resolution_clock::now()}
    {}
    ~ScopeTimer()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        std::cout << "ScopeTimer: " << name << " takes " << deltaTime << " milliseconds.\n";
    }
    std::string name;
    std::chrono::steady_clock::time_point startTime;
};