#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include<optix_device.h>

#include"material/material_definition.h"
#include"utils/Params.h"
#include"shaders/Cook-Torrance.h"
#include"shaders/shader_common.h"
extern "C" {
    __constant__ Params::Params params;
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}
static __forceinline__ __device__ float3 getPayload()
{
    float3 p;
    p.x=__uint_as_float(optixGetPayload_0());
    p.y= __uint_as_float(optixGetPayload_1());
    p.z= __uint_as_float(optixGetPayload_2());
    return p;
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
    const unsigned int image_index = idx.y * params.image_width + idx.x;
    unsigned int seed = GPURAND::tea<4>(idx.y * w + idx.x, idx.x);
    unsigned int seed2 = seed + 1;

    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 16; i++)
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(GPURAND::rng(seed), GPURAND::rng(seed2));

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
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2);
        result += make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    }

    result /= 16.0f;
    // Record results in our output raster
    params.image[image_index] = Shader_COM:: make_color(result);
}

extern "C" __global__ void __miss__ms() {
    Params::MissData* miss_data = reinterpret_cast<Params::MissData*>(optixGetSbtDataPointer());
    setPayload(miss_data->bg_color);
}

extern "C" __global__ void __closesthit__ch() {
    Material::material_Info* mat = reinterpret_cast<Material::material_Info*>(optixGetSbtDataPointer());
    Material::intersection its(mat->vertexData);
    float3 wo = -optixGetWorldRayDirection();
    float3 wi;
    float pdf;
    float3 f = CookTor::SampleCookTor_f(its, wo, wi, mat, pdf);
    unsigned int p0=0, p1=0, p2=0;
    optixTrace(
        params.handle,
        its.hit_pos+0.0001f*wi,//新光线的起点
        wi,                     //新光线的方向
        0.0f,                // Min intersection distance
        1e16f,               // Max intersection distance
        0.0f,                // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        1,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1, p2);
    float3 result = f * getPayload() * dot(wi, its.normal);
    setPayload(result);
}