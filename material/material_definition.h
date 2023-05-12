#pragma once
#include"shaders/shader_common.h"
#include"sutil/vec_math.h"
#include<cuda_runtime.h>
#include<optix_stubs.h>
#include <sutil/Exception.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include<nanovdb/util/IO.h>
#include<nanovdb/util/CudaDeviceBuffer.h>
#include<nanovdb/util/SampleFromVoxels.h>
#include"utils/Params.h"

/*���ͷ�ļ����������˲��ʵĶ��壬��β�������˸��ֲ��ʶ�Ӧ��SBTRecord*/
namespace Material {
	struct baseData {
		float3* vertices;
		uint3* indices;
		float3* normals;
		//float3* tangents;
		//float3* bitangents;
		float2* texcoords;
	};

	struct material_Info {
		//Ŀǰֻ�ܶ�ȡobj�ļ�����obj�ļ���û��pbr�Ĳ���
		material_Info() {}
		material_Info(const baseData& data, aiMaterial* aiMat)
		{
			vertexData = data;
			if (aiMat->GetTextureCount(aiTextureType_DIFFUSE) == 0) {
				//û����ͼ��ֱ�Ӷ�ȡģ�͵�basecolor��ֵ
				aiColor3D color;
				aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
				albedo = make_float3(color.r, color.g, color.b);
			}
			else
			{
				aiString tex_str;
				int iw, ih, n;
				aiMat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_str);
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


				CUDA_CHECK(cudaCreateTextureObject(&(this->albedo_tex), &texRes, &texDesc, nullptr));
				CUDA_SYNC_CHECK();
			}
			//��ȡ������

		}
		baseData vertexData;
		bool isLight;
		float3 lightColor = make_float3(1.0f,1.0f,1.0f);
		float3 albedo;
		cudaTextureObject_t albedo_tex= NULL;
		float metallic = 1.0f;
		float roughness = 1.0f;
		float ao = 0.0f;
		float HgPhase_g = 0.0f;
	};
	
	struct disney_material {
		disney_material() {}
		disney_material(const baseData& data, aiMaterial* aiMat)
		{
			vertexData = data;
			if (aiMat->GetTextureCount(aiTextureType_DIFFUSE) == 0) {
				//û����ͼ��ֱ�Ӷ�ȡģ�͵�basecolor��ֵ
				aiColor3D color;
				aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
				baseColor = make_float3(color.r, color.g, color.b);
			}
			else
			{
				aiString tex_str;
				int iw, ih, n;
				aiMat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_str);
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


				CUDA_CHECK(cudaCreateTextureObject(&(this->basecolor_tex), &texRes, &texDesc, nullptr));
				CUDA_SYNC_CHECK();
			}
			//��ȡ������

		}
		baseData vertexData;
		bool isLight=false;
		float3 lightColor = make_float3(1.0f, 1.0f, 1.0f);
		//������ʿ��ԭ��brdf�Ĳ���
		//���¸�����Ĭ��ֵ��blender���ԭ��brdfĬ��ֵ����
		float3 baseColor;//����ɫ
		//����������ͼ�������ɫ����ͼָ�����ں����ļ���������2����������⴦��
		cudaTextureObject_t basecolor_tex = NULL;
		//�����ȣ��涨�����Ϊ0������Ϊ1��
		//��ֵ����1ʱ��������������ʣ�ǿ�����淴��ǿ�ȣ�ͬʱ���淴���𽥸����Ͻ���ɫ
		//�뵼�����������⣬��������ʹ�ð뵼�����Ч��
		float metallic = 0.0f;
		//�α��棬������������״
		float subsurface = 0.0f;
		//�߹�ǿ��(���淴��ǿ��)
		//���ƾ��淴���ռ�����ı��ʣ�����ȡ��������
		float specular = 0.8f;
		//�ֲڶȣ�Ӱ��������;��淴�� 
		float roughness = 0.1f;
		//�߹�Ⱦɫ����baseColorһ�𣬿��ƾ��淴�����ɫ
		//ע�⣬���Ƿ�����Ч���������侵�淴����Ȼ�Ƿǲ�ɫ
		float specularTint = 0;
		//�������Գ̶ȣ����ƾ��淴���ڲ�ͬ�����ϵ�ǿ�ȣ��Ⱦ��淴��߹���ݺ��
		//�涨��ȫ����ͬ��ʱΪ0����ȫ��������ʱΪ1
		float anisotropic = 0.0f;
		//����ȣ�һ�ֶ�������������һ�����ڲ���������������µĹ���  
		float sheen = 0.0f;
		//����ɫ������sheen����ɫ
		float sheenTint = 0.5;
		//����ǿ�ȣ����Ƶڶ������淴�䲨���γɼ���Ӱ�췶Χ
		float clearcoat = 0.0f;
		//�������ȣ�����͸��Ϳ��ĸ߹�ǿ�ȣ�����ȣ�
		//�涨����(satin)Ϊ0������(gloss)Ϊ1��
		float clearcoatGloss = 0.03f;
	};
	struct volume_mat {
		//���캯��
		volume_mat() {}
		volume_mat(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& gridHandle)
		{
			//��ȡդ������
			cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
			cudaStreamCreate(&stream);
			gridHandle.deviceUpload(stream, true);
			deviceGrid = gridHandle.deviceGrid<float>();
			if (deviceGrid == nullptr)
			{
				printf("deviceGrid is null\n");
			}
			else
			{
				printf("deviceGrid is not null\n");
			}
			//Trismpler=new nanovdb::SampleFromVoxels<nanovdb::DefaultReadAccessor<float>, 1, true>(deviceGrid->getAccessor());
			//auto temp= nanovdb::createSampler<1>(deviceGrid->getAccessor());
			//temp.sample();
			//Trismpler = &temp;
			auto cpuGrid = gridHandle.grid<float>();
			//��ȡդ�����ݵ������Сֵ
			cpuGrid->tree().extrema(this->min, this->max);
			printf("min=%f,max=%f\n", min, max);
		}
		nanovdb::NanoGrid<float>* deviceGrid=nullptr;
		float min;
		float max;
		nanovdb::SampleFromVoxels<nanovdb::DefaultReadAccessor<float>, 1, true>* Trismpler;
		float3 color = make_float3(1.0f, 0.0f, 1.0f);
	};
}

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<Params::RayGenData>     RayGenSbtRecord;
typedef SbtRecord<Params::MissData>       MissSbtRecord;
typedef SbtRecord<Params::HitGroupData>   HitGroupSbtRecord;
typedef SbtRecord<Material::material_Info>MaterialInfoSbtRecord;
typedef SbtRecord<Material::disney_material>DisneySbtRecord;
typedef SbtRecord<Material::volume_mat>    VolumeSbtRecord;