
#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include <fstream>
#include<string>
#include<vector>
namespace Utils {
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
}
struct Params
{
    uchar4* image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
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
//template<class T>
//struct TriangleMeshSBTData {
//    float2* texcoord;
//    glm::vec3* vertices;
//    uint3* indices;
//    glm::vec3* normals;
//    cudaTextureObject_t texture;
//    __device__ float4 add(const float4& data) { data + make_float4(0.5, 0.0f, 0.0f, 0.0f); }
//    //__noinline__ __device__ float4 tex2dfloat4(const cudaTextureObject_t& t, float u, float v) { return  tex2D<float4>(texture, u, v); }
//    float4 ret_color2(T index, cudaTextureObject_t texture, float u, float v);
//    //{
//    //    float2 uv = (1.0f - u - v) * this->texcoord[index.x] + u * this->texcoord[index.y] + v * this->texcoord[index.z];
//    //    float4 fcolor2 = tex2dfloat4(texture, uv.x, uv.y);
//    //    //float4 fcolor2 = add(make_float4(0.3f));
//    //    return fcolor2;
//    //}
//
//};
// 
// 
//struct testSBTData {
//    //float2* texcoord;
////glm::vec3* vertices;
////uint3* indices;
//    //glm::vec3* normals;
//    cudaTextureObject_t texture;
//    uint3 tes1;
//    uint3 tes2;
//};

float4 readCudaTex4(cudaTextureObject_t texture, float u, float v);