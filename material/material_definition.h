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

/*这个头文件不仅包括了材质的定义，结尾还包括了各种材质对应的SBTRecord*/
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
		//目前只能读取obj文件，而obj文件并没有pbr的参数
		material_Info() {}
		material_Info(const baseData& data, aiMaterial* aiMat)
		{
			vertexData = data;
			if (aiMat->GetTextureCount(aiTextureType_DIFFUSE) == 0) {
				//没有贴图，直接读取模型的basecolor的值
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
			//读取金属度

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
				//没有贴图，直接读取模型的basecolor的值
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
			//读取金属度

		}
		baseData vertexData;
		bool isLight=false;
		float3 lightColor = make_float3(1.0f, 1.0f, 1.0f);
		//描述迪士尼原则brdf的参数
		//以下各属性默认值按blender里的原理brdf默认值设置
		float3 baseColor;//基础色
		//若物体有贴图，则基础色由贴图指定，在后续的计算会针对这2种情况做特殊处理
		cudaTextureObject_t basecolor_tex = NULL;
		//金属度，规定电介质为0，金属为1；
		//当值趋向1时：弱化漫反射比率，强化镜面反射强度，同时镜面反射逐渐附带上金属色
		//半导体材质情况特殊，尽量避免使用半导体调试效果
		float metallic = 0.0f;
		//次表面，控制漫反射形状
		float subsurface = 0.0f;
		//高光强度(镜面反射强度)
		//控制镜面反射光占入射光的比率，用于取代折射率
		float specular = 0.8f;
		//粗糙度，影响漫反射和镜面反射 
		float roughness = 0.1f;
		//高光染色，和baseColor一起，控制镜面反射的颜色
		//注意，这是非物理效果，且掠射镜面反射依然是非彩色
		float specularTint = 0;
		//各向异性程度，控制镜面反射在不同朝向上的强度，既镜面反射高光的纵横比
		//规定完全各向同性时为0，完全各项异性时为1
		float anisotropic = 0.0f;
		//光泽度，一种额外的掠射分量，一般用于补偿布料在掠射角下的光能  
		float sheen = 0.0f;
		//光泽色，控制sheen的颜色
		float sheenTint = 0.5;
		//清漆强度，控制第二个镜面反射波瓣形成及其影响范围
		float clearcoat = 0.0f;
		//清漆光泽度，控制透明涂层的高光强度（光泽度）
		//规定缎面(satin)为0，光泽(gloss)为1；
		float clearcoatGloss = 0.03f;
	};
	struct volume_mat {
		//构造函数
		volume_mat() {}
		volume_mat(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& gridHandle)
		{
			//获取栅格数据
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
			//获取栅格数据的最大最小值
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