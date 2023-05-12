# 设置cmake最小版本
cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE "D:/utils/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(VCPKG_TARGET_TRIPLET "x64-windows")

# 设置项目名称
project(OPTIX-Render LANGUAGES CXX CUDA)

# 引入cuda
find_package(CUDAToolkit REQUIRED)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# 设置cuda标准
SET(CMAKE_CUDA_STANDARD 20)

# 设置源文件列表
list(APPEND SOURCE_FILES
        source.cpp
)
set(exename optix-render)

# 添加可执行文件
add_executable(${exename} ${SOURCE_FILES})

target_include_directories(${exename} PUBLIC D:/nvidia_optix_7_6_0/include)
target_include_directories(${exename} PUBLIC D:/nvidia_optix_7_6_0/SDK)
target_include_directories(${exename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/buffers)
target_include_directories(${exename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/geometry)
target_include_directories(${exename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/material)
target_include_directories(${exename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pipline)
target_include_directories(${exename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ray)
target_include_directories(${exename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
target_include_directories(${exename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_include_directories(${exename} PUBLIC "D:/cuda/CUDA Development/include")

# 包含nanovdb头文件
target_include_directories(${exename} PUBLIC D:/utils/vcpkg/vcpkg/buildtrees/openvdb/src/3ab5951686-1c7a23646f.clean/nanovdb)

target_link_libraries(${exename} PUBLIC D:/nvidia_optix_7_6_0/build/lib/Debug/sutil_7_sdk.lib)
target_link_libraries(${exename} PUBLIC D:/nvidia_optix_7_6_0/build/lib/Release/sutil_7_sdk.lib)
target_link_libraries(${exename} PUBLIC "D:/cuda/CUDA Development/lib/x64/cudart_static.lib")

# 链接第三方库
find_package(glad CONFIG REQUIRED)
target_link_libraries(${exename} PRIVATE glad::glad)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${exename} PRIVATE glfw)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${exename} PRIVATE glm::glm)

find_package(assimp CONFIG REQUIRED)
target_link_libraries(${exename} PRIVATE assimp::assimp)

find_package(RapidJSON CONFIG REQUIRED)
target_link_libraries(${exename} PRIVATE rapidjson)

find_package(Stb REQUIRED)
target_include_directories(${exename} PRIVATE ${Stb_INCLUDE_DIR})

# 找到第三方包openvdb
find_package(OpenVDB CONFIG REQUIRED)

# 打印
message(STATUS "OpenVDB    config: ${OpenVDB_DIR}")
message(STATUS "OpenVDB    version: ${OpenVDB_VERSION}")
message(STATUS "OpenVDB    include: ${OpenVDB_INCLUDE_DIRS}")
message(STATUS "OpenVDB    libraries: ${OpenVDB_LIBS}")

# 链接openvdb库
target_link_libraries(${exename} PRIVATE OpenVDB::openvdb)
