//model由多个mesh组成，因为assimp是按照一个mesh一个材质来分
//所以这里也这么做，一个mesh只会对应一种材质(assimp会自动根据材质不同分配mesh)
// 另外每个mesh里设置了material_Info用来给optix传SBT Record
// 目前采取的方法是所有材质参数都放进这个material_Info里(其实都堆一个材质里不太好，这样做方便)
//
#pragma once
#include<iostream>
#include<format>
#include<vector>

#include <cuda_runtime.h>
#include <sutil/Exception.h>

#include"buffer/CUDABuffer.h"
#include"material/material_definition.h"
#include"../pipline/shaders.h"
namespace Gemotery {
	//template<class T>
	//concept materialAble = requires(T mat, Ray::HitData data, aiMaterial * aiMat)
	//{
	//	mat(data, aiMat);
	//};

	//template<class mat>
	//requires materialAble<mat>
	class assimp_mesh {
	public:
		assimp_mesh(const aiMesh* mesh, const aiScene* aiscene, unsigned ID);
		void setLight(const float3& lightColor) { this->matInfo.isLight = true; this->matInfo.lightColor = lightColor; }
		void drawHit(Shaders::hitShaders& hitShader);
		//host端
		std::vector<float3> v;//顶点坐标
		std::vector<float3> vn;//顶点法线
		std::vector<float2>  vt;//顶点纹理坐标
		std::vector<float3>  tangents;//顶点切线
		std::vector<float3>  bitangents;//顶点副切线
		std::vector<uint3>  indices;//顶点索引
		Material::disney_material matInfo;
		//device端
		CUdeviceptr d_vertice;//单独设置一个这个是因为triangle_input[count].triangleArray.vertexBuffers
		//要的是CUDeviceptr的地址，必须要保存为左值
		UtilsBuffer::CUDABuffer d_verticesBuffer;
		UtilsBuffer::CUDABuffer d_normalBuffer;
		UtilsBuffer::CUDABuffer d_texcoordBuffer;
		UtilsBuffer::CUDABuffer d_tangentBuffer;
		UtilsBuffer::CUDABuffer d_biTangentBuffer;
		UtilsBuffer::CUDABuffer d_indicesBuffer;
		//因为一个model有可能有多个网格，所以给每个模型中的所有网格标号
		unsigned int assimp_model_ID;
		std::string mesh_name;
		unsigned face_num;
		unsigned vertice_num;
		unsigned indice_num;
	};

	//template<class mat>
	//	requires materialAble<mat>
	 //inline assimp_mesh<mat>::assimp_mesh(const aiMesh* mesh, const aiScene* aiscene, unsigned ID) {
	inline assimp_mesh::assimp_mesh(const aiMesh* mesh, const aiScene* aiscene, unsigned ID) {
		assimp_model_ID = ID;
		mesh_name = mesh->mName.C_Str();
		vertice_num = mesh->mNumVertices;
		face_num = mesh->mNumFaces;

		//处理顶点数据，包括顶点位置，纹理坐标，法线，切线，副切线
		for (unsigned int i = 0; i < vertice_num; i++) {
			v.push_back(make_float3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
			vn.push_back(make_float3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
			tangents.push_back(make_float3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z));
			bitangents.push_back(make_float3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z));
			if (mesh->mTextureCoords[0]) // 网格是否有纹理坐标？这里用[0]是因为一个顶点可以有多个纹理坐标
			{
				vt.push_back(make_float2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y));
			}
		}
		std::cout << std::format("网格 {}:\n", mesh_name);
		std::cout << std::format("网格编号 {}:\n", assimp_model_ID);
		std::cout << std::format("顶点数目 {}:\n", v.size());
		std::cout << std::format("模型面数 {}:\n\n", face_num);
		//处理索引
		for (unsigned int i = 0; i < face_num; i++)
		{
			aiFace face = mesh->mFaces[i];
			indices.push_back(make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
		}
		indice_num = indices.size();
		std::cout << std::format("索引数目(按照有几个面) {}:\n\n", indice_num);
		//把host端数据复制到device端
		d_verticesBuffer.alloc_and_upload(v);
		d_vertice = d_verticesBuffer.d_pointer();//单独设置一个这个是因为triangle_input[count].triangleArray.vertexBuffers
		//要的是CUDeviceptr的地址，必须要保存为左值
		d_normalBuffer.alloc_and_upload(vn);
		d_texcoordBuffer.alloc_and_upload(vt);
		d_tangentBuffer.alloc_and_upload(tangents);
		d_biTangentBuffer.alloc_and_upload(bitangents);
		d_indicesBuffer.alloc_and_upload(indices);

		//处理材质
		Material::baseData hitdata;
		hitdata.vertices = (float3*)d_verticesBuffer.d_pointer();
		hitdata.indices = (uint3*)d_indicesBuffer.d_pointer();
		hitdata.normals = (float3*)d_normalBuffer.d_pointer();
		//hitdata.tangents = (float3*)d_tangentBuffer.d_pointer();
		//hitdata.bitangents = (float3*)d_biTangentBuffer.d_pointer();
		hitdata.texcoords = (float2*)d_texcoordBuffer.d_pointer();

		if (mesh->mMaterialIndex > 0)
		{
			aiMaterial* mat = aiscene->mMaterials[mesh->mMaterialIndex];
			matInfo = Material::disney_material(hitdata, mat);
		}
	}

	inline void assimp_mesh::drawHit(Shaders::hitShaders& hitShader)
	{

	}

}