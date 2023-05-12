/*
* Cook-TorranceBRDF��ʵ�֣�����G���D������ò���GGXģ��
* ʵ�ֲο�learnOpengl��Pbr����
* ��Ҫ�Բ����Ƕ�D��(CGX)��Ҫ�Բ�������https://zhuanlan.zhihu.com/p/57032810
* ��ʱ�ò���ao
* �Ը�brdf����Ҫ�Բ����ĸĽ��ɼ�https://schuttejoe.github.io/post/ggximportancesamplingpart1/
*/
#pragma once
#include"intersection.h"
#include"rand_gen.h"
#include"shader_common.h"
namespace CookTor {
    //F��  ( Fresnel Equation ) ���������̣��������������������ڲ�ͬ�ı�����±���������Ĺ�����ռ�ı��ʡ�
    //������ý��Ʋ���ԭ����
    __forceinline__ __device__ float3 fresnelSchlick(float cosTheta, float3 F0)
    {
        return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
    }
    //D�� ( Normal Distribution Function ) ���߷ֲ��������������ܵ�����ֲڶȵ�Ӱ���£�ȡ�������м�����һ�µ�΢ƽ���������������������΢ƽ�����Ҫ������
    __forceinline__ __device__ float DistributionGGX(float3 N, float3 H, float roughness)
    {
        float a = roughness * roughness+0.0001f;
        float a2 = a * a;
        float NdotH = max(dot(N, H), 0.0);
        float NdotH2 = NdotH * NdotH;

        float nom = a2;
        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = M_PIf * denom * denom;

        return nom / denom;
    }

    __forceinline__ __device__ float GeometrySchlickGGX(float NdotV, float roughness)
    {
        float r = (roughness + 1.0);
        float k = (r * r) / 8.0;

        float nom = NdotV;
        float denom = NdotV * (1.0 - k) + k;

        return nom / denom;
    }

    //G��( Geometry Function ) ���κ�����������΢ƽ���Գ���Ӱ�����ԡ���һ��ƽ����ԱȽϴֲڵ�ʱ��ƽ������ϵ�΢ƽ���п��ܵ�ס������΢ƽ��Ӷ����ٱ���������Ĺ��ߡ�
    __forceinline__ __device__ float GeometrySchlickGGX(float NdotV, float NdotL,float roughness)
    {
        float tNdotV = fmaxf(NdotV, 0.0);
        float tNdotL = fmaxf(NdotL, 0.0);
        float ggx2 = GeometrySchlickGGX(tNdotV, roughness);
        float ggx1 = GeometrySchlickGGX(tNdotL, roughness);

        return ggx1 * ggx2;
    }

    __forceinline__ __device__ float CookTorPDF(const Material::intersection& its,const float3& wo, const float3& wi,  Material::material_Info *mat) {
        float3 wh = normalize(wo + wi);
        return DistributionGGX(its.normal,wh, mat->roughness)*dot(wh,its.normal) / (4.0f*dot(wo,wh));
    }

     __forceinline__ __device__ float3 evalCookTorBXDF(const Material::intersection & its, const float3 & wo, const float3 & wi,  Material::material_Info *mat) {
        float3 albedoColor;
        if (mat->albedo_tex != NULL) {
            float4 temp= tex2D<float4>(mat->albedo_tex, its.uv_coord.x, its.uv_coord.y);
            albedoColor = make_float3(temp.x, temp.y, temp.z);
        }
        else {
            albedoColor = mat->albedo;
        }
        float3 N = its.normal;
        float3 H = normalize(wi + wo);
        float3 V = wo;
        float3 L = wi;
        float3 F0 = make_float3(0.04f);
        F0 = lerp(F0, albedoColor, mat->metallic);

        //float NDF = DistributionGGX(N, H, mat->roughness);
        float2 a2 = make_float2(mat->roughness);
        float3 lH = its.tbn.transformToLocal(H);
        float NDF = Shader_COM::ggx_D_pdf(a2, lH).x;
        float G = GeometrySchlickGGX(dot(N,V), dot(N,L),mat->roughness);
        float3 F = fresnelSchlick(max(dot(H, L), 0.0f), F0);

        float3 kS = F;
        float3 kD = make_float3(1.0f) - kS;
        kD *= 1.0f - mat->metallic;

        float3 nominator = NDF * G * F;
        float denominator = 4.0f * max(dot(N, V), 0.0f) * max(dot(N, L), 0.0f) + 0.001f;
        float3 specular = nominator / denominator;

        return (kD * albedoColor /M_PIf+ specular);
        //return albedoColor;
    }
     //����CGX���ɲ���������ȡ��pdfֵ������brdfֵ
     __forceinline__ __device__ float3 SampleCookTor_f(const Material::intersection& its, const float3& wo, float3& wi,  Material::material_Info *mat, float& pdf) {
         //���ɲ�������
         const uint3  idx = optixGetLaunchIndex();
         float2 rand2 = make_float2((curand_uniform(its.state), curand_uniform(its.state2)));
         float alpha2 = mat->roughness* mat->roughness;
         float cosTheta2 = max(0.0f,(1 - rand2.x) / (rand2.x * (alpha2 - 1) + 1));
         float cosTheta = (0.0f,sqrt(cosTheta2));
         float sinTheta = (0.0f,sqrt(1 - cosTheta2));
         float phi = 2 * M_PIf * rand2.y;
         //float3 lh=make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
         float3 lh = Shader_COM::sampleHemisphere(rand2);
         float2 a2=make_float2(alpha2,alpha2);
         //float3 lh = Shader_COM::ggx_sample(a2,rand2);
         float3 lo = its.tbn.transformToLocal(wo);
         float3 li = reflect(-lo,lh);
         float3 wh = its.tbn.transformToWorld(lh);
         //wi = reflect(-wo,wh);
         wi = its.tbn.transformToWorld(li);
         //�ж����ɵĲ��������Ƿ��ڵ�ǰ�����İ�������
         //if ((li.z < 0.0f && dot(wi, its.normal) > 0) || (li.z > 0.0f && dot(wi, its.normal) < 0)) {
         //    pdf = 0;
         //    return make_float3(1.0f, 0.0f, 0.0f);
         //}
         //return lo;
         if (li.z < 0.0f) {
             pdf = 0.0;
             return make_float3(0.0f, 0.0f, 0.0f);
         }
         // if (dot(wi,its.normal)<0){
         //    pdf = 0;
         //    return make_float3(0.0f,0.0f,1.0f);
         //}
         //����pdf
         //pdf = CookTorPDF(its,wo,wi,mat) ;
         pdf = 0.5*M_1_PIf;
         //pdf = Shader_COM::ggx_D_pdf(a2,lh).y/(4.0f*dot(wh,wo));
         //����bxdf
         return evalCookTorBXDF(its,wo,wi,mat);
     }
}