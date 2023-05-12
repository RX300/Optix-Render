#pragma once
#include"ray.h"
#include"optix.h"
#include"optix_device.h"
namespace Ray{
	__device__ ray_intersection::ray_intersection(const HitData& hit) {
		const float2 barycentrics = optixGetTriangleBarycentrics();
		unsigned int primIdx = optixGetPrimitiveIndex();
		this->prim_id = primIdx;
		glm::uvec3 index = hit.indices[primIdx];
		//��ֵuv
		this->uv_coord = (1.0f - barycentrics.x - barycentrics.y) * hit.texcoords[index.x] +
			barycentrics.x * hit.texcoords[index.y] + barycentrics.y * hit.texcoords[index.z];
		//��ֵhit_pos
		this->hit_pos = (1.0f - barycentrics.x - barycentrics.y) * hit.vertices[index.x] +
			barycentrics.x * hit.vertices[index.y] + barycentrics.y * hit.vertices[index.z];
		//��ֵnormal
		this->normal = (1.0f - barycentrics.x - barycentrics.y) * hit.normals[index.x] +
			barycentrics.x * hit.normals[index.y] + barycentrics.y * hit.normals[index.z];
		//��ֵtangent
		this->tangent = (1.0f - barycentrics.x - barycentrics.y) * hit.tangents[index.x] +
			barycentrics.x * hit.tangents[index.y] + barycentrics.y * hit.tangents[index.z];
		//��ֵbitangent
		this->bitangent = (1.0f - barycentrics.x - barycentrics.y) * hit.bitangents[index.x] +
			barycentrics.x * hit.bitangents[index.y] + barycentrics.y * hit.bitangents[index.z];
		//�ж��Ƿ���б���
		if (optixIsBackFaceHit())
		{
			normal = -normal;
			bitangent = -bitangent;
		}
}
}