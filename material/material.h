#pragma once
#include <cuda_runtime.h>

#include"glm/glm.hpp"
#include"ray/ray.h"
#include"utils/utils.h"


#include<optix_stubs.h>
#include <sutil/Exception.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

namespace Material {
	enum materialType  { testIndex };
    class base_material {
    public:
		  base_material() {}
		  base_material(const Ray::HitData& data, aiMaterial* mat);
          unsigned getBxdfType()const;
		  __forceinline__ __device__ glm::vec3 evalBxdf(const glm::vec3& wi, const glm::vec3& wo, const Ray::ray_intersection& its);
           glm::vec3 bxdfSample(glm::vec3& wi, const glm::vec3& wo, float& pdf, const Ray::ray_intersection& its) { return glm::vec3(0); }
           float bxdfPdf(const glm::vec3& wi, const glm::vec3& wo, const Ray::ray_intersection& its) { return 0.0f; }
    
        Ray::HitData hitData;
        glm::vec3 baseColor;
        cudaTextureObject_t baseTexture = NULL;
    };
#pragma region
	  base_material::base_material(const Ray::HitData& data, aiMaterial* mat){
		          hitData = data;
	  	if (mat->GetTextureCount(aiTextureType_DIFFUSE) == 0) {
	  		//没有贴图，直接读取模型的basecolor的值
	  		aiColor3D color;
	  		mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
	  		baseColor = glm::vec3(color.r, color.g, color.b);
	  	}
	  	else
	  	{
	  		aiString tex_str;
	  		int iw, ih, n;
	  		mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_str);
	  		unsigned char* data = stbi_load(
	  			tex_str.C_Str(), &iw, &ih, &n, 4);
	  		uint32_t* pixel = (uint32_t*)data;
	  		cudaChannelFormatDesc channelDesc =
	  			cudaCreateChannelDesc<uchar4>();
	  		cudaArray_t cuArray;
	  		CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, iw, ih));
	  		size_t bytesPerElem = sizeof(unsigned char) * 4;
	  		CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, data, iw * bytesPerElem,
	  			iw * bytesPerElem, ih, cudaMemcpyHostToDevice));

	  		cudaTextureDesc texDesc;
	  		memset(&texDesc, 0, sizeof(texDesc));
	  		texDesc.addressMode[0] = cudaAddressModeClamp;
	  		texDesc.addressMode[1] = cudaAddressModeClamp;
	  		texDesc.filterMode = cudaFilterModeLinear;
	  		texDesc.readMode = cudaReadModeNormalizedFloat;
	  		texDesc.normalizedCoords = 1;


	  		cudaResourceDesc texRes;
	  		memset(&texRes, 0, sizeof(cudaResourceDesc));

	  		texRes.resType = cudaResourceTypeArray;
	  		texRes.res.array.array = cuArray;


	  		CUDA_CHECK(cudaCreateTextureObject(&(this->baseTexture), &texRes, &texDesc, nullptr));
	  		CUDA_SYNC_CHECK();
	  	}
	}
#pragma endregion
#pragma region
    struct test_materialData {
		test_materialData() {}
		test_materialData(const Ray::HitData& data, glm::vec3 color, cudaTextureObject_t tex)
        {
            hitData = data;
            baseColor = color;
            baseTexture = tex;
        }
		test_materialData(const Ray::HitData& data, aiMaterial* mat) {
			hitData = data;
			if (mat->GetTextureCount(aiTextureType_DIFFUSE) == 0) {
				//没有贴图，直接读取模型的basecolor的值
				aiColor3D color;
				mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
				baseColor = glm::vec3(color.r, color.g, color.b);
			}
			else
			{
				aiString tex_str;
				int iw, ih, n;
				mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_str);
				unsigned char* data = stbi_load(
					tex_str.C_Str(), &iw, &ih, &n, 4);
				uint32_t* pixel = (uint32_t*)data;
				cudaChannelFormatDesc channelDesc =
					cudaCreateChannelDesc<uchar4>();
				cudaArray_t cuArray;
				CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, iw, ih));
				size_t bytesPerElem = sizeof(unsigned char) * 4;
				CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, data, iw * bytesPerElem,
					iw * bytesPerElem, ih, cudaMemcpyHostToDevice));

				cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = cudaAddressModeClamp;
				texDesc.addressMode[1] = cudaAddressModeClamp;
				texDesc.filterMode = cudaFilterModeLinear;
				texDesc.readMode = cudaReadModeNormalizedFloat;
				texDesc.normalizedCoords = 1;


				cudaResourceDesc texRes;
				memset(&texRes, 0, sizeof(cudaResourceDesc));

				texRes.resType = cudaResourceTypeArray;
				texRes.res.array.array = cuArray;


				(cudaCreateTextureObject(&(this->baseTexture), &texRes, &texDesc, nullptr));
				CUDA_SYNC_CHECK();
			}
		}

		materialType type = materialType::testIndex;
        Ray::HitData hitData;
        glm::vec3 baseColor;
        cudaTextureObject_t baseTexture = NULL;
    };
#pragma endregion
}