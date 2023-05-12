//model�ɶ��mesh��ɣ���Ϊassimp�ǰ���һ��meshһ����������
//��������Ҳ��ô����һ��meshֻ���Ӧһ�ֲ���(assimp���Զ����ݲ��ʲ�ͬ����mesh)
// ����ÿ��mesh��������material_Info������optix��SBT Record
// Ŀǰ��ȡ�ķ��������в��ʲ������Ž����material_Info��(��ʵ����һ�������ﲻ̫�ã�����������)
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
		//host��
		std::vector<float3> v;//��������
		std::vector<float3> vn;//���㷨��
		std::vector<float2>  vt;//������������
		std::vector<float3>  tangents;//��������
		std::vector<float3>  bitangents;//���㸱����
		std::vector<uint3>  indices;//��������
		Material::disney_material matInfo;
		//device��
		CUdeviceptr d_vertice;//��������һ���������Ϊtriangle_input[count].triangleArray.vertexBuffers
		//Ҫ����CUDeviceptr�ĵ�ַ������Ҫ����Ϊ��ֵ
		UtilsBuffer::CUDABuffer d_verticesBuffer;
		UtilsBuffer::CUDABuffer d_normalBuffer;
		UtilsBuffer::CUDABuffer d_texcoordBuffer;
		UtilsBuffer::CUDABuffer d_tangentBuffer;
		UtilsBuffer::CUDABuffer d_biTangentBuffer;
		UtilsBuffer::CUDABuffer d_indicesBuffer;
		//��Ϊһ��model�п����ж���������Ը�ÿ��ģ���е�����������
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

		//���������ݣ���������λ�ã��������꣬���ߣ����ߣ�������
		for (unsigned int i = 0; i < vertice_num; i++) {
			v.push_back(make_float3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
			vn.push_back(make_float3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
			tangents.push_back(make_float3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z));
			bitangents.push_back(make_float3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z));
			if (mesh->mTextureCoords[0]) // �����Ƿ����������ꣿ������[0]����Ϊһ����������ж����������
			{
				vt.push_back(make_float2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y));
			}
		}
		std::cout << std::format("���� {}:\n", mesh_name);
		std::cout << std::format("������ {}:\n", assimp_model_ID);
		std::cout << std::format("������Ŀ {}:\n", v.size());
		std::cout << std::format("ģ������ {}:\n\n", face_num);
		//��������
		for (unsigned int i = 0; i < face_num; i++)
		{
			aiFace face = mesh->mFaces[i];
			indices.push_back(make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
		}
		indice_num = indices.size();
		std::cout << std::format("������Ŀ(�����м�����) {}:\n\n", indice_num);
		//��host�����ݸ��Ƶ�device��
		d_verticesBuffer.alloc_and_upload(v);
		d_vertice = d_verticesBuffer.d_pointer();//��������һ���������Ϊtriangle_input[count].triangleArray.vertexBuffers
		//Ҫ����CUDeviceptr�ĵ�ַ������Ҫ����Ϊ��ֵ
		d_normalBuffer.alloc_and_upload(vn);
		d_texcoordBuffer.alloc_and_upload(vt);
		d_tangentBuffer.alloc_and_upload(tangents);
		d_biTangentBuffer.alloc_and_upload(bitangents);
		d_indicesBuffer.alloc_and_upload(indices);

		//�������
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