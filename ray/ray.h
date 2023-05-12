#pragma once
#include"glm/glm.hpp"
#include <cuda_runtime.h>
namespace Ray{
    struct HitData {
        glm::vec3* vertices;
        glm::uvec3* indices;
        glm::vec3* normals;
        glm::vec3* tangents;
        glm::vec3* bitangents;
        glm::vec2* texcoords;
    };
    enum IntersectionAttribute {
        Intersection_Normal = 0x01,
        Intersection_Tangent = 0x02,
        Intersection_Bitangent = 0x04,
        Intersection_UV = 0x08
    };
    class ray_intersection {
    public:
         __device__ ray_intersection() {}
         __device__ ray_intersection(const HitData& hit);
        //传进ray，并且利用ray里的参数得到击中点的uv值，并且利用uv值算出其他属性的插值。
        __forceinline__ __device__ glm::vec3 getNormal()const { return normal; }
        __forceinline__ __device__ glm::vec3 getTangent()const { return tangent; }
        __forceinline__ __device__ glm::vec3 getBitangent()const { return bitangent; }
        __forceinline__ __device__ glm::vec2 getUvCoord()const { return uv_coord; }
        __forceinline__ __device__ glm::vec3 getHitPos()const { return hit_pos; }


        unsigned int prim_id;
        glm::vec3 hit_pos;
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec3 bitangent;
        glm::vec2 uv_coord;
    };
}