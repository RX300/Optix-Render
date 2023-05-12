//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <glm/fwd.hpp> 
#include"glm/glm.hpp"
#include <optix.h>

#include"utils/utils_gpu.h"
#include <cuda/helpers.h>
#include"material_gpu_impl.h"
#include"Ray/ray_impl.h"

#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include<iostream>
#include"curand.h"
//#include <internal/optix_7_device_impl.h>


extern "C" {
    __constant__ Params params;
}


static  __forceinline__ __device__ float4 ret_color(TriangleMeshSBTData<uint3>& sbt_data, uint3 index, cudaTextureObject_t texture, float u, float v) {
    float2 uv = (1.0f - u - v) * sbt_data.texcoord[index.x] + u * sbt_data.texcoord[index.y] + v * sbt_data.texcoord[index.z];
    float4 fcolor2 = tex2D<float4>(texture, uv.x, uv.y);
    return fcolor2;
}

static  __forceinline__ __device__ float4 ret_color2(Material::test_materialData& sbt_data, uint3 index, cudaTextureObject_t texture, float u, float v) {
    //glm::vec2 uv = (1.0f - u - v) * sbt_data.hitData.texcoords[index.x] + u * sbt_data.hitData.texcoords[index.y] + v * sbt_data.hitData.texcoords[index.z];
    float4 fcolor2 = tex2D<float4>(texture, u, v);
    return fcolor2;
}

static  __forceinline__ __device__ float3 comnormal(Material::base_material& sbt_data, uint3 index, float u, float v) {
    glm::vec3 uv = (1.0f - u - v) * sbt_data.hitData.normals[index.x] + u * sbt_data.hitData.normals[index.y] + v * sbt_data.hitData.normals[index.z];
    return make_float3(-uv.x, -uv.y, -uv.z);
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}
static __forceinline__ __device__ glm::vec3 getPayload()
{
    return glm::vec3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()));
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)
    ) - 1.0f;

    origin = params.cam_eye;
    direction = normalize(d.x * U + d.y * V + W);
}
template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}
// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}
// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int& prev)
{
    return ((float)lcg(prev) / (float)0x01000000);
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const int    w = params.image_width;
    const int    h = params.image_height;
    const float3 eye = params.cam_eye;
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const uint3  idx = optixGetLaunchIndex();
    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.image_width + launch_index.x;
    unsigned int seed = tea<4>(idx.y * w + idx.x, idx.x);
    unsigned int seed2 = seed + 1;

    float3 result=make_float3(0.0f);
    for (int i = 0; i < 16; i++)
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed2));

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
        ) - 1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        // Trace the ray against our scene hierarchy
        unsigned int p0, p1, p2;
        optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            2,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2);
        result += make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    }
    
    result /= 16.0f;
    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color(result);
}
extern "C" __global__ void __miss__shadow()
{

}

extern "C" __global__ void __miss__ms()
{
    MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    float3 test = miss_data->bg_color;
    test.x = 0.0f;
    setPayload(test);
}
extern "C" __global__ void __closesthit__shadow()
{
    setPayload(make_float3(0.0f, 0.0f, 0.0f));
}
extern "C" __global__ void __anyhit__ah()
{

}
extern "C" __global__ void __anyhit__shadow()
{
    //setPayload(make_float3(0.0f, 0.0f, 0.0f));
    //optixTerminateRay();
}

extern "C" __global__ void __closesthit__ch()
{

    const float2 barycentrics = optixGetTriangleBarycentrics();
    Material::base_material* mat_data = reinterpret_cast<Material::base_material*>(optixGetSbtDataPointer());
    Ray::ray_intersection its(mat_data->hitData);
    glm::vec3 color = mat_data->evalBxdf(glm::vec3(0), glm::vec3(0),its);
    float3 origin = optixGetWorldRayOrigin();
    float3 dir = optixGetWorldRayDirection();
    float t = optixGetRayTmax();
    float3 hit = origin + t * dir;
    glm::vec3 glmhit = glm::vec3(hit.x,hit.y,hit.z);
    glm::vec3 ray_origin = its.hit_pos;
    glm::vec3 light_pos(4.0f, 1.0f, 5.0f);
    glm::vec3 light_dir = glm::normalize(light_pos - glmhit);
    glm::vec3 hit_origin = glmhit + 0.001f * light_dir;

    glm::vec3 cur = getPayload();
    unsigned int p0 = __float_as_uint(color.x);
    unsigned int p1 = __float_as_uint(color.y);
    unsigned int p2 = __float_as_uint(color.z);
    optixTrace(
        params.handle,
        make_float3(hit_origin.x, hit_origin.y, hit_origin.z),
        make_float3(light_dir.x, light_dir.y, light_dir.z) ,
        0.0f,                // Min intersection distance
        1e16f,               // Max intersection distance
        0.0f,                // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        1,                   // SBT offset   -- See SBT discussion
        2,                   // SBT stride   -- See SBT discussion
        1,                   // missSBTIndex -- See SBT discussion
        p0, p1, p2);

    setPayload(make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2)));

}
