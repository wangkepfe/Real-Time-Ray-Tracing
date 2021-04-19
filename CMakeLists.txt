cmake_minimum_required(VERSION 3.10)
project(myproject LANGUAGES CUDA CXX)

file(GLOB mySRC
    "src/*.h"
    "src/*.cpp"
    "src/*.cuh"
    "src/*.cu"
    "ext/TinyTIFF/tinytiffreader.cpp"
    "ext/TinyTIFF/tinytiffreader.h"
    "ext/stb/stb_image.h")

add_executable(myproject ${mySRC})

file(GLOB toolSrc
    "tool/meshProcessor.cpp"
    "src/*.h"
    "src/*.cuh")

add_executable(meshProcesor ${toolSrc})
target_include_directories(meshProcesor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# c++
set_target_properties(myproject PROPERTIES CXX_STANDARD 17)
add_definitions(-DNOMINMAX)

# cuda
target_include_directories(myproject PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/cuda-samples/Common")
target_include_directories(meshProcesor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/cuda-samples/Common")
target_link_libraries(myproject cudadevrt.lib)
set_target_properties(myproject PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(myproject PRIVATE "--keep")

# vulkan
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Vulkan REQUIRED)
target_compile_definitions(myproject PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_include_directories(myproject PRIVATE Vulkan::Vulkan)
target_link_libraries(myproject Vulkan::Vulkan)

# glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw")
target_link_libraries(myproject glfw)

# assimp
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/assimp")
target_link_libraries(myproject assimp)
target_link_libraries(meshProcesor assimp)

# header only
target_include_directories(myproject PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/stb")
target_include_directories(myproject PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/TinyTIFF")

# copy resource
add_custom_target(copy_resources)
FILE(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/resources" resources_source)
FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/resources" resources_destination)
add_custom_command(
    TARGET copy_resources POST_BUILD
    COMMAND rd ${resources_destination} /s /q
    COMMAND mklink /D ${resources_destination} ${resources_source})
add_dependencies(myproject copy_resources)