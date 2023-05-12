#pragma once
#include"material/material_definition.h"
#include<optix_device.h>
namespace Material {
    class intersection {
    public:
        __host__ __device__ intersection() {}
		__host__ __device__ intersection(const Material::baseData& hit, curandState* State, curandState* State2);
        //传进ray，并且利用ray里的参数得到击中点的uv值，并且利用uv值算出其他属性的插值。
		curandState* state;
		curandState* state2;
		unsigned int seed;
        unsigned int prim_id;
        float3 hit_pos;
        float3 normal;
		Shader_COM::TBN tbn;
        float2 uv_coord;
		float3 tangent;
		float3 bitangent;
    };

    intersection::intersection(const Material::baseData& hit,curandState* State, curandState* State2)
    {
		this->state = State; this->state2 = State2;
		const float2 barycentrics = optixGetTriangleBarycentrics();
		unsigned int primIdx = optixGetPrimitiveIndex();
		this->prim_id = primIdx;
		uint3 index = hit.indices[primIdx];
		//插值uv
		this->uv_coord = (1.0f - barycentrics.x - barycentrics.y) * hit.texcoords[index.x] +
			barycentrics.x * hit.texcoords[index.y] + barycentrics.y * hit.texcoords[index.z];
		//插值hit_pos
		this->hit_pos = (1.0f - barycentrics.x - barycentrics.y) * hit.vertices[index.x] +
			barycentrics.x * hit.vertices[index.y] + barycentrics.y * hit.vertices[index.z];
		//插值normal
		this->normal = (1.0f - barycentrics.x - barycentrics.y) * hit.normals[index.x] +
			barycentrics.x * hit.normals[index.y] + barycentrics.y * hit.normals[index.z];
		////插值tangent
		//this->tangent = (1.0f - barycentrics.x - barycentrics.y) * hit.tangents[index.x] +
		//	barycentrics.x * hit.tangents[index.y] + barycentrics.y * hit.tangents[index.z];
		////插值bitangent
		//this->bitangent = (1.0f - barycentrics.x - barycentrics.y) * hit.bitangents[index.x] +
		//	barycentrics.x * hit.bitangents[index.y] + barycentrics.y * hit.bitangents[index.z];
		//判断是否击中背面
		if (optixIsBackFaceHit())
		{
			normal = -normal;
			//bitangent = -bitangent;
		}
		tbn.computeTBN(normal);
		tangent = tbn.tangent;
		bitangent = tbn.bitangent;
    }
}