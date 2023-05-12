#pragma once
#include"material/material_definition.h"
#include<optix_device.h>
namespace Material {
    class intersection {
    public:
        __host__ __device__ intersection() {}
		__host__ __device__ intersection(const Material::baseData& hit, curandState* State, curandState* State2);
        //����ray����������ray��Ĳ����õ����е��uvֵ����������uvֵ����������ԵĲ�ֵ��
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
		//��ֵuv
		this->uv_coord = (1.0f - barycentrics.x - barycentrics.y) * hit.texcoords[index.x] +
			barycentrics.x * hit.texcoords[index.y] + barycentrics.y * hit.texcoords[index.z];
		//��ֵhit_pos
		this->hit_pos = (1.0f - barycentrics.x - barycentrics.y) * hit.vertices[index.x] +
			barycentrics.x * hit.vertices[index.y] + barycentrics.y * hit.vertices[index.z];
		//��ֵnormal
		this->normal = (1.0f - barycentrics.x - barycentrics.y) * hit.normals[index.x] +
			barycentrics.x * hit.normals[index.y] + barycentrics.y * hit.normals[index.z];
		////��ֵtangent
		//this->tangent = (1.0f - barycentrics.x - barycentrics.y) * hit.tangents[index.x] +
		//	barycentrics.x * hit.tangents[index.y] + barycentrics.y * hit.tangents[index.z];
		////��ֵbitangent
		//this->bitangent = (1.0f - barycentrics.x - barycentrics.y) * hit.bitangents[index.x] +
		//	barycentrics.x * hit.bitangents[index.y] + barycentrics.y * hit.bitangents[index.z];
		//�ж��Ƿ���б���
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