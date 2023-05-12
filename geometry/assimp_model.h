#pragma once
#include"gasBuild.h"
#include"assimp_mesh.h"
namespace Gemotery {
	class assimp_model:public OptixGas::gasHitObject {
	public:
		assimp_model(const std::string& pathname);
		~assimp_model()
		{
			for (auto &i : meshes)
			{
				delete i;
			}
		}
	virtual void inputGasData( std::vector<OptixBuildInput>&) override;
	void drawHit(Shaders::hitShaders& hitShader);
	OptixTraversableHandle buildGAS(const OptixDeviceContext& optixContext);
	void setLight(const unsigned int& meshIndex, const float3& lightColor=make_float3(1.0f)) { meshes[meshIndex]->setLight(lightColor); }
	int getMeshNum() { return meshes.size(); }
	
		void processNode( aiNode* node, const aiScene* aiscene);
		assimp_mesh* processMesh( aiMesh* mesh, const aiScene* aiscene);

		std::vector<assimp_mesh*>meshes;
		UtilsBuffer::CUDABuffer asBuffer;

	};
	inline assimp_model::assimp_model(const std::string& pathname) {
		Assimp::Importer import;
		const aiScene* scene = import.ReadFile(pathname, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace
			| aiProcess_GenUVCoords
		);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
			return;
		}
		printf("模型信息:\n 名字:%s\n", pathname.c_str());
		processNode( scene->mRootNode, scene);
	}
	inline void assimp_model::inputGasData(std::vector<OptixBuildInput>&Builds)
	{
		const uint32_t triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
		for (auto& cur_mesh : this->meshes){
			OptixBuildInput curbuild{};
			curbuild.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			curbuild.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			curbuild.triangleArray.numVertices = static_cast<unsigned int>(cur_mesh->vertice_num);
			curbuild.triangleArray.vertexBuffers = &cur_mesh->d_vertice;
			curbuild.triangleArray.flags = triangle_input_flags;

			curbuild.triangleArray.vertexStrideInBytes = 0;
			curbuild.triangleArray.numSbtRecords = 1;
			curbuild.triangleArray.sbtIndexOffsetBuffer = 0;
			curbuild.triangleArray.sbtIndexOffsetSizeInBytes = 0;
			curbuild.triangleArray.sbtIndexOffsetStrideInBytes = 0;

			curbuild.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			curbuild.triangleArray.indexBuffer = cur_mesh->d_indicesBuffer.d_pointer();
			curbuild.triangleArray.numIndexTriplets = static_cast<unsigned int>(cur_mesh->indice_num);
			curbuild.triangleArray.indexStrideInBytes = sizeof(uint3);
			Builds.push_back(curbuild);
		}
	}
	inline void assimp_model::drawHit(Shaders::hitShaders& hitShader) {
		for (auto& cur_mesh : this->meshes) {
			cur_mesh->drawHit(hitShader);
		}
	}
	inline OptixTraversableHandle assimp_model::buildGAS(const OptixDeviceContext& optixContext) {


		// ==================================================================
		// triangle inputs
		// ==================================================================
		const uint32_t triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
		std::vector<OptixBuildInput>triangle_input(meshes.size());

		unsigned int count = 0;
		for (auto &cur_mesh : this->meshes) {
			triangle_input[count].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input[count].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input[count].triangleArray.numVertices = static_cast<unsigned int>(cur_mesh->vertice_num);
			triangle_input[count].triangleArray.vertexBuffers = &cur_mesh->d_vertice;
			triangle_input[count].triangleArray.flags = triangle_input_flags;

			triangle_input[count].triangleArray.vertexStrideInBytes = 0;
			triangle_input[count].triangleArray.numSbtRecords = 1;
			triangle_input[count].triangleArray.sbtIndexOffsetBuffer = 0;
			triangle_input[count].triangleArray.sbtIndexOffsetSizeInBytes = 0;
			triangle_input[count].triangleArray.sbtIndexOffsetStrideInBytes = 0;

			triangle_input[count].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangle_input[count].triangleArray.indexBuffer = cur_mesh->d_indicesBuffer.d_pointer();
			triangle_input[count].triangleArray.numIndexTriplets = static_cast<unsigned int>(cur_mesh->indice_num);
			triangle_input[count].triangleArray.indexStrideInBytes = sizeof(uint3);

			count++;

		}   

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    //CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    //CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );


		// ==================================================================
		// BLAS setup
		// ==================================================================
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
			| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
			;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage
		(optixContext,
			&accelOptions,
			triangle_input.data(),
			(int)meshes.size(),  // num_build_inputs
			&blasBufferSizes
		));    

		// ==================================================================
		// prepare compaction
		// ==================================================================
		UtilsBuffer::CUDABuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.d_pointer();    
		// ==================================================================
		// execute build (main stage)
		// ==================================================================
		UtilsBuffer::CUDABuffer tempBuffer;
		tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

		UtilsBuffer::CUDABuffer outputBuffer;
		outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

		OptixTraversableHandle asHandle{ 0 };
		OPTIX_CHECK(optixAccelBuild(optixContext,
			/* stream */0,
			&accelOptions,
			triangle_input.data(),
			(int)meshes.size(),
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,

			outputBuffer.d_pointer(),
			outputBuffer.sizeInBytes,

			&asHandle,

			&emitDesc, 1
		));
		CUDA_SYNC_CHECK();    
		// ==================================================================
		// perform compaction
		// ==================================================================
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);


		this->asBuffer.alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixContext,
			/*stream:*/0,
			asHandle,
			asBuffer.d_pointer(),
			asBuffer.sizeInBytes,
			&asHandle));
		CUDA_SYNC_CHECK();    
		// ==================================================================
		//  clean up
		// ==================================================================
		outputBuffer.free(); // << the UNcompacted, temporary output buffer
		tempBuffer.free();
		compactedSizeBuffer.free();
		

		return asHandle;
	}
	inline void assimp_model::processNode(aiNode* node, const aiScene* aiscene) {
		// 处理节点所有的网格（如果有的话）
		for (unsigned int i = 0; i < node->mNumMeshes; i++)
		{
			aiMesh* mesh = aiscene->mMeshes[node->mMeshes[i]];
			assimp_mesh* ass_mesh = processMesh( mesh, aiscene);
			meshes.push_back(ass_mesh);
		}
		// 接下来对它的子节点重复这一过程
		for (unsigned int i = 0; i < node->mNumChildren; i++)
		{
			processNode( node->mChildren[i], aiscene);
		}
	}
	inline assimp_mesh* assimp_model::processMesh(aiMesh* mesh, const aiScene* aiscene) {
		assimp_mesh* mesh_p = new assimp_mesh( mesh, aiscene, meshes.size());
		//Assimp规定一个网格对象只能对应一个材质对象，如果源模型一个网格有多个材质对象
		return mesh_p;
	}
}