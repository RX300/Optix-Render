#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include<optix_device.h>

#include"utils/Params.h"
#include"shaders/Disney-Brdf.h"
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include"shaders/device_vdb.h"

__device__ unsigned long long int count = 0;
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
    unsigned spp = 1;
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
    unsigned long long int piexel_num = w * h;
    //atomicAdd(&count, (unsigned long long int)1);
    float progress = (float)count / (float)piexel_num;
    //printf("complete\n");
}

extern "C" __global__ void __miss__ms() {
    Params::MissData* miss_data = reinterpret_cast<Params::MissData*>(optixGetSbtDataPointer());
    setPayload(miss_data->bg_color);
}

extern "C" __global__ void __intersection__volume()
{
    const auto* sbt_data = reinterpret_cast<const Material::volume_mat*>(optixGetSbtDataPointer());
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(
        sbt_data->deviceGrid);
    assert(grid);

    // compute intersection points with the volume's bounds in index (object) space.
    const float3 ray_orig = optixGetObjectRayOrigin();
    const float3 ray_dir = optixGetObjectRayDirection();

    auto bbox = grid->indexBBox();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    auto iRay = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(ray_orig),
        reinterpret_cast<const nanovdb::Vec3f&>(ray_dir), t0, t1);

    if (iRay.intersects(bbox, t0, t1))
    {
        // report the exit point via payload
        optixSetPayload_0(__float_as_uint(t1));
        // report the entry-point as hit-point
        optixReportIntersection(fmaxf(t0, optixGetRayTmin()), 0);
    }
}

//template<class T>__forceinline__ __device__ T TriSample(const nanovdb::NanoGrid<T>* deviceGrid, nanovdb::Vec3f worldpos) {
//    auto gpuAcc = deviceGrid->getAccessor();
//    // 推断gpuAcc的类型,并用这个类型实例化sampler（TrilinearSampler）
//    nanovdb::SampleFromVoxels<decltype(gpuAcc), 1, true> sampler(gpuAcc);
//    auto xyz = deviceGrid->worldToIndex(worldpos);
//    return sampler(xyz);
//}
__forceinline__ __device__ float TriSample(const nanovdb::NanoGrid<float>* deviceGrid,nanovdb::DefaultReadAccessor<float>&gpuAcc, nanovdb::Vec3f worldpos) {
    nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(deviceGrid->worldToIndex(worldpos));
    return gpuAcc.getValue(ijk);
}
using TriSampleFromVoxels=nanovdb::SampleFromVoxels<nanovdb::DefaultReadAccessor<float>, 1, true>;
__forceinline__ __device__ float TriSample2(const nanovdb::NanoGrid<float>* deviceGrid, TriSampleFromVoxels& Trismpler,nanovdb::Vec3f worldpos) {
   
    auto xyz = deviceGrid->worldToIndex(worldpos);
    return (Trismpler)(xyz);
}

__forceinline__ __device__ bool is_intersection__volume(const nanovdb::Ray<float>& ray, const nanovdb::NanoGrid<float>* volumeGrid, float& exitT)
{
    float t0 = ray.t0();
    float t1 = ray.t1();
    auto bbox = volumeGrid->indexBBox();
    if (ray.intersects(bbox, t0, t1))
    {
        // report the exit point via payload
        exitT = t1;
        return true;
    }
    else
        return false;
}
__forceinline__ __device__ float3 delta_tracking_sampling(nanovdb::Ray<float>& ray, nanovdb::DefaultReadAccessor<float>& gpuAcc,curandState* state, curandState* state2, const nanovdb::NanoGrid<float>* volumeGrid, TriSampleFromVoxels& Trismpler,float maxDesity, bool& is_inMedium, float extinctionCrossSection)
{
    float2 rand2 = make_float2(curand_uniform(state), curand_uniform(state2));
    float t = 0;
    float density;
    float transfer_density;
    float maxDensity = DeviceVdb::transferCTA(maxDesity).w;
    float majorant = maxDensity * extinctionCrossSection;
    float tmax = 0;
    float4 transfer_color;
    bool flag=is_intersection__volume(ray, volumeGrid, tmax);
    while (flag)
    {

        float eplison = curand_uniform(state);
        t += -log(1 - eplison) / (majorant);
        if (t > tmax)
        {
            is_inMedium = false;
            break;
        }
        auto pos = ray(t);
        density = TriSample2(volumeGrid, Trismpler,pos);
        transfer_color = DeviceVdb::transferCTA(density);
        transfer_density = transfer_color.w;
        if (transfer_density / maxDensity > float(curand_uniform(state)))
        {
            float3 dir = Shader_COM::sampleSphere(rand2);
            ray = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(pos),
                reinterpret_cast<const nanovdb::Vec3f&>(dir), 0.000f, 9999.0f);
            //glm::vec3 wi;
            //HG_Sample(-ray.direction(), &wi, Common::random2(0.0, 1.0));
            //ray = Ray::ray(ray.at(t), wi);
            return make_float3(transfer_color);

        }
    }
    //返回背景光
    return make_float3(1.0f);
}
extern "C" __global__ void __closesthit__ch() {
    //获取寄存器信息
    unsigned int seed = optixGetPayload_4();
    unsigned int depth = optixGetPayload_3();
    //获取相关体数据
    Material::volume_mat* v_data = reinterpret_cast<Material::volume_mat*>(optixGetSbtDataPointer());
    auto* volumeGrid= v_data->deviceGrid;
    auto gpuAcc = volumeGrid->tree().getAccessor();
    auto smp = nanovdb::createSampler<1>(gpuAcc);
    float max = v_data->max;
    auto state = &params.states[seed];
    auto state2= &params.states[seed + params.image_width * params.image_height];
    //获取光线信息
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    float3 hit_pos = ray_orig + t1 * ray_dir;
    float3 ray_orig2 = hit_pos + 0.0001f * ray_dir;
    //创建光线
    auto iRay = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(ray_orig2),
        reinterpret_cast<const nanovdb::Vec3f&>(ray_dir), 0.000f, 9999.0f);
    //保证光线弹射不超过深度上限
    if (depth < 5)
    {
        float3 beta = make_float3(1.0);
        bool is = true;
        unsigned int count = 0;
        //delta tracking最多进行3次
        while (is && count < 10) {
            beta *= delta_tracking_sampling(iRay, gpuAcc,state, state2,volumeGrid, smp, max, is,50.0f);
            count++;
        }
        setPayload(beta);
       
    }
    else
        setPayload(make_float3(0.0f, 0.0f, 0.0f));
}