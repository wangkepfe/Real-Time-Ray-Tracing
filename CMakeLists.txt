cmake_minimum_required(VERSION 3.10)
project(myproject LANGUAGES CUDA CXX)

file(GLOB mySRC
    "src/*.h"
    "src/*.cpp"
    "src/*.cuh"
    "src/*.cu"
    "ext/HDRloader/*.cpp"
    "ext/HDRloader/*.h")

add_executable(myproject ${mySRC})

# standard
set_target_properties(myproject PROPERTIES CXX_STANDARD 17)
add_definitions(-DNOMINMAX)

# cuda
target_include_directories(myproject PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common")
target_link_libraries(myproject curand.lib)
target_link_libraries(myproject cudadevrt.lib)
set_target_properties(myproject PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/glfw")
target_link_libraries(myproject glfw)

# externel source
target_include_directories(myproject PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ext/HDRloader")

# copy resource
add_custom_target(resources)
FILE(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/resources" source)
FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/resources" destination)
IF (WIN32)
add_custom_command(
TARGET resources POST_BUILD
COMMAND mklink /D ${destination} ${source}
DEPENDS ${destination}
COMMENT "symbolic link resources folder from ${source} => ${destination}"
)
ELSE()
add_custom_command(
TARGET resources POST_BUILD
COMMAND ${CMAKE_COMMAND} -E create_symlink ${source} ${destination}
DEPENDS ${destination}
COMMENT "symbolic link resources folder from ${source} => ${destination}"
)
ENDIF()
add_dependencies(myproject resources)