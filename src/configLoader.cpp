#include "configLoader.h"
#include "globalSettings.h"
#include <toml.hpp>

void LoadConfig(GlobalSettings* g_settings)
{
    auto data                        = toml::parse                            ("resources/config.toml");

    const auto resolution            = toml::find_or                          (data, "resolution", {});
    const auto file                  = toml::find_or                          (data, "file", {});
    const auto optimziation          = toml::find_or                          (data, "optimziation", {});

    g_settings->width                 = toml::find_or<int>                     (resolution, "width", 1920);
    g_settings->height                = toml::find_or<int>                     (resolution, "height", 1080);

    g_settings->inputMeshFileName     = toml::find_or<std::string>             (file, "inputMeshFileName", {});
    g_settings->inputTextureFileNames = toml::find_or<std::vector<std::string>>(file, "inputTextureFileNames", {});
    g_settings->inputCameraFileName   = toml::find_or<std::string>             (file, "inputCameraFileName", {});
    g_settings->cameraSaveFileName    = toml::find_or<std::string>             (file, "cameraSaveFileName", {});
    g_settings->loadCameraAtInit      = toml::find_or<bool>                    (file, "loadCameraAtInit", false);

    g_settings->useDynamicResolution  = toml::find_or<bool>                    (optimziation, "useDynamicResolution", true);
    g_settings->targetFps             = toml::find_or<float>                   (optimziation, "targetFps", 60.0);
    g_settings->maxWidth              = toml::find_or<int>                     (optimziation, "maxWidth", 3840);
    g_settings->maxHeight             = toml::find_or<int>                     (optimziation, "maxHeight", 2160);
    g_settings->minWidth              = toml::find_or<int>                     (optimziation, "minWidth", 640);
    g_settings->minHeight             = toml::find_or<int>                     (optimziation, "minHeight", 480);
}