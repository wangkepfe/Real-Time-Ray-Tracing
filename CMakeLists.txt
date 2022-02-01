cmake_minimum_required(VERSION 3.10)
project(realtimeraytracing LANGUAGES CUDA CXX)

file(GLOB mySRC
    "src/*.h"
    "src/*.cpp"
    "src/*.cuh"
    "src/*.cu"
    "ext/imgui/imconfig.h"
    "ext/imgui/imgui.h"
    "ext/imgui/imgui.cpp"
    "ext/imgui/imgui_internal.h"
    "ext/imgui/imgui_widgets.cpp"
    "ext/imgui/imgui_tables.cpp"
    "ext/imgui/imgui_draw.cpp"
    "ext/imgui/backends/imgui_impl_glfw.cpp"
    "ext/imgui/backends/imgui_impl_glfw.h"
    "ext/imgui/backends/imgui_impl_vulkan.cpp"
    "ext/imgui/backends/imgui_impl_vulkan.h"
    "ext/TinyTIFF/tinytiffreader.cpp"
    "ext/TinyTIFF/tinytiffreader.h"
    "ext/stb/stb_image.h")

add_executable(realtimeraytracing ${mySRC})

file(GLOB toolSrc
    "tool/meshProcessor.cpp"
    "src/*.h"
    "src/*.cuh")

add_executable(meshProcesor ${toolSrc})
target_include_directories(meshProcesor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# c++
set_target_properties(realtimeraytracing PROPERTIES CXX_STANDARD 20)
add_definitions(-DNOMINMAX)

# cuda
target_include_directories(realtimeraytracing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/cuda-samples/Common")
target_include_directories(meshProcesor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/cuda-samples/Common")
target_link_libraries(realtimeraytracing cudadevrt.lib)
set_target_properties(realtimeraytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(realtimeraytracing PRIVATE "--keep")

# vulkan
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Vulkan REQUIRED)
target_compile_definitions(realtimeraytracing PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_include_directories(realtimeraytracing PRIVATE Vulkan::Vulkan)
target_link_libraries(realtimeraytracing Vulkan::Vulkan)

# glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw")
target_link_libraries(realtimeraytracing glfw)

# assimp
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/assimp")
target_link_libraries(realtimeraytracing assimp)
target_link_libraries(meshProcesor assimp)

# header only
target_include_directories(realtimeraytracing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/stb")
target_include_directories(realtimeraytracing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/TinyTIFF")
target_include_directories(realtimeraytracing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/toml")
target_include_directories(realtimeraytracing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui")
target_include_directories(realtimeraytracing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/backends")

# copy resource
add_custom_target(copy_resources)
FILE(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/resources" resources_source)
FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/resources" resources_destination)
add_custom_command(
    TARGET copy_resources POST_BUILD
    COMMAND rd ${resources_destination} /s /q
    COMMAND mklink /D ${resources_destination} ${resources_source})
add_dependencies(realtimeraytracing copy_resources)