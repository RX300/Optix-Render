#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include<optix_device.h>

#include"utils/Params.h"
#include"shaders/Disney-Brdf.h"

//设置整个流水线都会用到的参数，这里固定extern"C"加__constant__
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
    //这里直接让种子等于image_index
    const unsigned int image_index = idx.y * params.image_width + idx.x;
    const unsigned int image_res = w * h;
    curand_init(image_index, 0, 0, &params.states[image_index]);
    curand_init(image_index, 1, 0, &params.states[image_index+ image_res]);
    unsigned spp = 1000;
    float3 result =make_float3(0.0f);
    for (int i = 0; i < spp; i++)
    {
        const float2 subpixel_jitter = make_float2((curand_uniform(&params.states[image_index]), curand_uniform(&params.states[image_index + image_res])));

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
        ) - 1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        // Trace the ray against our scene hierarchy
        unsigned int p0, p1, p2,depth=0,seed= image_index;
        optixTrace(
            params.handle,
            ray_origin,//这里直接设置起点就行，但是后续需要往采样方向延伸
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2,depth,seed);
        result += make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
        //result += getPayload();
    }

    result /= (float)spp;
    // Record results in our output raster
    params.image[image_index] = Shader_COM:: make_color(result);
}

extern "C" __global__ void __miss__ms() {
    Params::MissData* miss_data = reinterpret_cast<Params::MissData*>(optixGetSbtDataPointer());
    setPayload(miss_data->bg_color);
}

extern "C" __global__ void __closesthit__ch() {
    unsigned int seed = optixGetPayload_4();
    unsigned int depth = optixGetPayload_3();
    Material::disney_material* mat = reinterpret_cast<Material::disney_material*>(optixGetSbtDataPointer());
    Material::intersection its(mat->vertexData, &params.states[seed], &params.states[seed + params.image_width * params.image_height]);
    if (mat->isLight == true)
    {
        setPayload(mat->lightColor);

    }
    else if (depth < 3)
    {
        float3 ray_orig = optixGetWorldRayOrigin();
        float3 ray_dir = optixGetWorldRayDirection();
        float t1 = optixGetRayTmax();
        float3 hit_pos = ray_orig + t1 * ray_dir;

        float3 wo = normalize( - optixGetWorldRayDirection());
        float3 wi;
        float pdf; 
        float2 rand2 = make_float2(curand_uniform(its.state), curand_uniform(its.state2));
        wi = its.tbn.transformToWorld(Shader_COM::sampleCosineWeightedHemisphere(rand2));
        float3 f = Disney_Brdf::bxdfSample(its, wo, wi, pdf, mat);
        //float3 f = Disney_Brdf::evalBxdf(its, wo, wi, mat);
        //pdf = dot(wi, its.normal) / M_PIf;
        float cos = dot(its.normal, wi);
        unsigned int p0, p1, p2 ;
        depth++;
        //optixTrace(
        //    params.handle,
        //    its.hit_pos+0.0001f*wi,//新光线的起点，记得往采样方向延伸一段
        //    wi,                     //新光线的方向
        //    0.0f,                // Min intersection distance
        //    1e16f,               // Max intersection distance
        //    0.0f,                // rayTime -- used for motion blur
        //    OptixVisibilityMask(255), // Specify always visible
        //    OPTIX_RAY_FLAG_NONE,
        //    0,                   // SBT offset   -- See SBT discussion
        //    1,                   // SBT stride   -- See SBT discussion
        //    0,                   // missSBTIndex -- See SBT discussion
        //    p0, p1, p2,depth);
        optixTrace(
            params.handle,
            hit_pos + 0.0001f * wi,//新光线的起点，记得往采样方向延伸一段
            wi,                     //新光线的方向
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2, depth);
        if(pdf<0.000001f)
            setPayload(make_float3(0.0f,0.0f,0.0f));
        else
            setPayload(make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2))*cos*f/pdf);
    }
    else
        setPayload(make_float3(0.0f, 0.0f, 0.0f));


        //unsigned int seed = optixGetPayload_4();
        //Material::disney_material* mat = reinterpret_cast<Material::disney_material*>(optixGetSbtDataPointer());
        //Material::intersection its(mat->vertexData,&params.states[seed], &params.states[seed+params.image_width* params.image_height]);
        //its.seed = seed;
        ////Material::intersection its(mat->vertexData, params.states);
        //float3 wo = ( -optixGetWorldRayDirection());
        //float3 wi;
        //float pdf; 
        //float3 f = Disney_Brdf::bxdfSample(its, wo, wi, pdf,mat);
        //float cos = dot(its.normal, wi);
        //if(pdf<0.000001f)
        //    setPayload(make_float3(0.0f,0.0f,0.0f));
        //else
        //setPayload(make_float3(1.0f)*cos*f/pdf);
}