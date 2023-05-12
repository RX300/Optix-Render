#pragma once
#include <optix.h>
#include<curand_kernel.h>
#include"shader_common.h"
#include<nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#include<nanovdb/util/CudaDeviceBuffer.h>
#include<nanovdb/util/SampleFromVoxels.h>
namespace DeviceVdb {
	template<class T>__forceinline__ __device__ T TriSample(const nanovdb::NanoGrid<T>* volumeGrid, nanovdb::Vec3f worldpos) {
		auto gpuAcc = volumeGrid->getAccessor();
		// 推断gpuAcc的类型,并用这个类型实例化sampler（TrilinearSampler）
		nanovdb::SampleFromVoxels<decltype(gpuAcc), 1, true> sampler(gpuAcc);
		auto xyz = volumeGrid->worldToIndex(worldpos);
		return sampler(xyz);
	}
	__forceinline__ __device__ void CoordinateSystem(const float3 v1, float3* v2,
		float3* v3) {
		if (abs(v1.x) > abs(v1.y))
			*v2 = make_float3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
		else
			*v2 = make_float3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
		*v3 = cross(v1, *v2);
	}
	__forceinline__ __device__ float3 SphericalDirection(float sinTheta, float cosTheta, float phi,
		const float3 x, const float3 y,
		const float3 z) {
		return sinTheta * cos(phi) * x + sinTheta * sin(phi) * y +
			cosTheta * z;
	}
	__forceinline__ __device__ float HG(float cosTheta,float g)
	{
		float denom = 1 + g * g + 2 * g * cosTheta;
		return Inv4Pi * (1 - g * g) / (denom * sqrt(denom));
	}
	// HenyeyGreenstein Method Definitions
	__forceinline__ __device__ float HG_Sample(const float3 wo, float3* wi,
		const const float2 u,float g) {
		// Compute $\cos \theta$ for Henyey--Greenstein sample
		float cosTheta;
		if (abs(g) < 1e-3)
			cosTheta = 1 - 2 * u.x;
		else {
			float sqrTerm = (1 - g * g) / (1 + g - 2 * g * u.x);
			cosTheta = -(1 + g * g - sqrTerm * sqrTerm) / (2 * g);
		}

		// Compute direction _wi_ for Henyey--Greenstein sample
		float sinTheta = sqrt(fmaxf(0.0f, 1 - cosTheta * cosTheta));
		float phi = 2 * M_PIf * u.y;
		float3 v1, v2;
		CoordinateSystem(wo, &v1, &v2);
		*wi = SphericalDirection(sinTheta, cosTheta, phi, v1, v2, wo);
		return HG(cosTheta,g);
	}
	__forceinline__ __device__ float4 transferCTA3(float density)
	{
		float d = density * 0.005f;
		if (d < 0.1f)
			return make_float4(0.0f, 0.0f, 1.0f, 0.0f);
		else if (d < 0.4f)
			return make_float4(0.5f, 0.0f, 0.0f, 0.0f);
		else if (d < 0.9f)
			return make_float4(0.515f, 0.143f, 0.084f, 0.f);
		else
			return make_float4(0.8f, 0.8f, 0.8f, 1.0f);
	}
	__forceinline__ __device__ float4 transferCTA(float density)
	{
		float d = density * 0.005;
		if (d < 0.4)
			return make_float4(0, 0, 1, 0);
		else if (d < 0.7)
			return make_float4(0.515f, 0.143f, 0.084f, 0.6f);
		else
			return make_float4(0.757f, 0.731f, 0.613f, 1.0f);
	}
	//template<class T>
	//__forceinline__ __device__ bool is_intersection__volume(const nanovdb::Ray<T>&ray, const nanovdb::NanoGrid<T>* volumeGrid,float & exitT)
	//{
	//	auto bbox = volumeGrid->indexBBox();
	//	if (ray.intersects(bbox, t0, t1))
	//	{
	//		// report the exit point via payload
	//		exitT = t1;
	//		return true;
	//	}
	//	else
	//		return false;
	//}
	//template<class T>
	//__forceinline__ __device__ float3 delta_tracking_sampling(nanovdb::Ray<T>& ray, curandState* state, curandState* state2, const nanovdb::NanoGrid<T>* volumeGrid, float maxDesity,bool is_inMedium,float extinctionCrossSection)
	//{
	//	float2 rand2 = make_float2(curand_uniform(state), curand_uniform(state2));
	//	float t = 0;
	//	float density;
	//	float transfer_density;
	//	float maxDensity = transferCTA3(maxDesity).w;
	//	float majorant = maxDensity * extinctionCrossSection;
	//	float tmax = 0;
	//	float4 transfer_color;
	//	if (scene->rayIntersection1(ray, its, 0))
	//	{
	//		tmax = ray.gettfar();
	//	}
	//	while (true)
	//	{

	//		float eplison = curand_uniform(state);
	//		t += -log(1 - eplison) / (majorant);
	//		if (t > tmax)
	//		{
	//			is_inMedium = false;
	//			break;
	//		}
	//		auto pos=ray(t);
	//		density=TriSample(volumeGrid, pos);
	//		transfer_color = transferCTA3(density);
	//		transfer_density = transfer_color.w;
	//		if (transfer_density / maxDensity > float(curand_uniform(state)))
	//		{
	//			ray = Ray::ray(ray.at(t), Common::random_in_unit_sphere());
	//			//glm::vec3 wi;
	//			//HG_Sample(-ray.direction(), &wi, Common::random2(0.0, 1.0));
	//			//ray = Ray::ray(ray.at(t), wi);
	//			return make_float3(transferCTA3);

	//		}
	//	}
	//	return glm::vec3(1.0f);
	//}
}