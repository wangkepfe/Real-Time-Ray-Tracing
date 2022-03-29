#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "ui.h"
#include "kernel.cuh"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

extern RayTracer* g_rayTracer;

void UpdateUI(GLFWwindow* window)
{
    ImGui::Begin("Settings");

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Camera pos=(%.2f, %.2f, %.2f)", g_rayTracer->GetCamera().pos.x, g_rayTracer->GetCamera().pos.y, g_rayTracer->GetCamera().pos.z);
    ImGui::Text("Camera dir=(%.2f, %.2f, %.2f)", g_rayTracer->GetCamera().dir.x, g_rayTracer->GetCamera().dir.y, g_rayTracer->GetCamera().dir.z);
    ImGui::Text("Triangle count = %d", g_rayTracer->GetTriangleCount());

    if (ImGui::CollapsingHeader("Render Pass", ImGuiTreeNodeFlags_None))
    {
        auto list = g_rayTracer->renderPassSettings.GetValueList();
        for (auto& itempair : list)
            ImGui::Checkbox(itempair.second.c_str(), itempair.first);
    }

    if (ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_None))
    {
        auto list = g_rayTracer->skyParams.GetValueList();
        for (auto& item : list)
        {
            if (std::get<2>(item) == UiWidgetType::Scalar)
            {
                if (ImGui::SliderFloat(std::get<1>(item).c_str(),
                                       std::get<0>(item),
                                       std::get<3>(item),
                                       std::get<4>(item),
                                       "%.3f",
                                       std::get<5>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None))
                {
                    g_rayTracer->skyParams.needRegenerate = true;
                }
            }
            else if (std::get<2>(item) == UiWidgetType::Input)
            {
                if (ImGui::InputFloat(std::get<1>(item).c_str(), std::get<0>(item)))
                {
                    g_rayTracer->skyParams.needRegenerate = true;
                }
            }
        }
    }

    if (ImGui::CollapsingHeader("Sampling", ImGuiTreeNodeFlags_None))
    {
        auto list = g_rayTracer->sampleParams.GetValueList();
        for (auto& item : list)
        {
            if (std::get<3>(item) == UiWidgetType::Scalar)
            {
                if (ImGui::SliderFloat(std::get<2>(item).c_str(),
                                       std::get<1>(item),
                                       std::get<4>(item),
                                       std::get<5>(item),
                                       "%.3f",
                                       std::get<6>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None))
                {}
            }
            else if (std::get<3>(item) == UiWidgetType::Input)
            {
                if (ImGui::InputFloat(std::get<2>(item).c_str(), std::get<1>(item)))
                {}
            }
            else if (std::get<3>(item) == UiWidgetType::Checkbox)
            {
                if (ImGui::Checkbox(std::get<2>(item).c_str(), std::get<0>(item)))
                {}
            }
        }
    }

    if (ImGui::CollapsingHeader("Tone Mapping", ImGuiTreeNodeFlags_None))
    {
        const char* items[] = { "Uncharted", "ACES1", "ACES2", "Reinhard" };
        ImGui::Combo("Tone Mapping Type", (int*)&g_rayTracer->postProcessParams.toneMappingType, items, IM_ARRAYSIZE(items));

        auto list = g_rayTracer->postProcessParams.GetValueList();
        for (auto& item : list)
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.2f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
    }

    if (ImGui::CollapsingHeader("Temporal Denoising", ImGuiTreeNodeFlags_None))
    {
        auto list = g_rayTracer->denoisingParams.GetValueList();
        for (auto& item : list)
        {
            if (ImGui::InputFloat(item.second.c_str(), item.first))
            {
                *item.first = max(*item.first, 0.00001f);
            }
        }
    }

    // if (ImGui::CollapsingHeader("Sky2", ImGuiTreeNodeFlags_None))
    // {
    //     if (ImGui::CollapsingHeader("Observer", ImGuiTreeNodeFlags_None))
    //     {
    //         ImGui::SliderFloat("Latitude", &g_rayTracer->skyParams.latitude, -90.0f, 90.0f);
    //         ImGui::SliderFloat("Longtitude", &g_rayTracer->skyParams.longtitude, -180.0f, 180.0f);
    //         ImGui::SliderFloat("Elevation", &g_rayTracer->skyParams.elevation, 0.0f, 2000.0f);
    //         ImGui::SliderInt("Month", &g_rayTracer->skyParams.month, 1, 12);
    //         ImGui::SliderInt("Day", &g_rayTracer->skyParams.day, 1, 31);
    //         ImGui::SliderFloat("Time of day", &g_rayTracer->skyParams.timeOfDay, 0.0f, 1.0f);
    //     }

    //     if (ImGui::CollapsingHeader("Quality", ImGuiTreeNodeFlags_None))
    //     {
    //         ImGui::SliderInt("Samples", &g_rayTracer->skyParams.numSamples, 1, 64);
    //         ImGui::SliderInt("Light Samples", &g_rayTracer->skyParams.numLightRaySamples, 1, 64);
    //     }

    //     if (ImGui::CollapsingHeader("Sun", ImGuiTreeNodeFlags_None))
    //     {
    //         ImGui::SliderFloat("Sun power", &g_rayTracer->skyParams.sunPower, 0.0f, 40.0f);
    //     }

    //     if (ImGui::CollapsingHeader("Earth", ImGuiTreeNodeFlags_None))
    //     {
    //         ImGui::SliderFloat("Earth radius (km)", &g_rayTracer->skyParams.earthRadius, 0.0f, 10000.0f);
    //         ImGui::SliderFloat("Atmosphere height (km)", &g_rayTracer->skyParams.atmosphereHeight, 0.0f, 1000.0f);
    //     }

    //     if (ImGui::CollapsingHeader("Rayleigh", ImGuiTreeNodeFlags_None))
    //     {
    //         ImGui::SliderFloat("Atmosphere thickness", &g_rayTracer->skyParams.atmosphereThickness, 0.0f, 40000.0f);
    //         ImGui::SliderFloat3("Rayleigh mean free path (km)", g_rayTracer->skyParams.mfpKmRayleigh._v, 0.0f, 1000.0f);
    //     }

    //     if (ImGui::CollapsingHeader("Mie", ImGuiTreeNodeFlags_None))
    //     {
    //         ImGui::SliderFloat("Aerosol thickness", &g_rayTracer->skyParams.aerosolThickness, 0.0f, 10000.0f);
    //         ImGui::SliderFloat("Mie mean free path (km)", &g_rayTracer->skyParams.mfpKmMie, 0.0f, 1000.0f);
    //         ImGui::SliderFloat("Mie albedo", &g_rayTracer->skyParams.albedoMie, 0.0f, 1.0f);
    //         ImGui::SliderFloat("g", &g_rayTracer->skyParams.g, 0.0f, 1.0f);
    //         const char* items[] = { "Henyey-Greenstein", "Mie" };
    //         ImGui::Combo("Phase function", (int*)&g_rayTracer->skyParams.miePhaseFuncType, items, IM_ARRAYSIZE(items));
    //     }
    // }

    if (ImGui::Button("Exit"))
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    ImGui::End();
}