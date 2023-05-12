#pragma once
#include<optix.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include<string>
#include<vector>
#include <curand.h>
#include <curand_kernel.h>
extern "C" void initOneState(curandState * devStates);
namespace Params {
    bool readSourceFile(std::string& str, const std::string& filename)
    {
        // Try to open file
        std::ifstream file(filename.c_str(), std::ios::binary);
        if (file.good())
        {
            // Found usable source file
            std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
            str.assign(buffer.begin(), buffer.end());
            return true;
        }
        return false;
    }

    void getPTXCode(std::string& ptx, const std::string& sourceFilePath)
    {
        if (!readSourceFile(ptx, sourceFilePath))
        {
            std::string err = "Couldn't open source file " + sourceFilePath;
            throw std::runtime_error(err.c_str());
        }
    }

    struct Params
    {
        uchar4* image;
        unsigned int           image_width;
        unsigned int           image_height;
        float3                 cam_eye;
        float3                 cam_u, cam_v, cam_w;
        OptixTraversableHandle handle;
        curandState* states;
    };


    struct RayGenData
    {
        // No data needed
    };


    struct MissData
    {
        float3 bg_color;
    };


    struct HitGroupData
    {
        float3 hit_color;
        // No data needed
        __forceinline__ __device__ float3 ret_color() { return make_float3(0.0f, 0.0f, 1.0f); }
    };
}