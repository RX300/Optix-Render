//shader�಻��shader�ļ������������ڹ���shader���࣬�����˴�
//ģ���ʼ��shader�Ĳ���(�������ó�����)������SBT Record��������
//������������ʱ��������gpu��SBT�������١�
// Ŀǰ��shader�಻�����쳣��������飬ֻ������4��������
// ��������ĳ��������úܶ���д���˵�Ҫ����ȷ��������Ҫ�ֶ���
// 
//������д������ˮ�ߵ������shader��
#pragma once
#include"buffer/CUDABuffer.h"
#include <optix.h>
#include <optix_stubs.h>
#include<vector>
#include<string>
#include"sutil/Exception.h"
#include "rapidjson/document.h"
#include"utils/Params.h"
#define IS_INDEX 0
#define AH_INDEX 1
#define CH_INDEX 2
#define DC_INDEX 0
#define CC_INDEX 1
//��Щshader�����й���vector�Ĳ�������ʹ�ù��캯����ʼ���󶼲��ܶ�vector����Ϊʹ��push_back��������������·����ڴ�(���Զ�ȡ����д��)
namespace Shaders {

	class rayGenShaders {
	public:rayGenShaders() {}
		  rayGenShaders(const OptixModule& module, const std::string& funcName, OptixDeviceContext context)
		  {
			  raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			  raygen_prog_group_desc.raygen.entryFunctionName = funcName.c_str();
			  raygen_prog_group_desc.raygen.module = module;

			  OPTIX_CHECK_LOG(optixProgramGroupCreate(
				  context,
				  &raygen_prog_group_desc,
				  1,   // num program groups
				  &raygen_group_options,
				  LOG, &LOG_SIZE,
				  &raygen_prog_group
			  ));
			  int i=1;
		  }
		  void load(const OptixModule& module, const std::string& funcName, OptixDeviceContext context)
		  {
			  raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			  raygen_prog_group_desc.raygen.entryFunctionName = funcName.c_str();
			  raygen_prog_group_desc.raygen.module = module;

			  OPTIX_CHECK_LOG(optixProgramGroupCreate(
				  context,
				  &raygen_prog_group_desc,
				  1,   // num program groups
				  &raygen_group_options,
				  LOG, &LOG_SIZE,
				  &raygen_prog_group
			  ));
			  int i = 1;
		  }
		  template<class T>
		  void setSbtRecord( std::vector<T>& SbtRecord, OptixShaderBindingTable& Sbt) {
			  OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, SbtRecord.data()));
			  RayGenBuffer.alloc_and_upload(SbtRecord);
			  Sbt.raygenRecord = RayGenBuffer.d_pointer();
		  }
		  ~rayGenShaders() { RayGenBuffer.free(); }
		OptixProgramGroup raygen_prog_group = {};
		OptixProgramGroupDesc raygen_prog_group_desc = {};
		OptixProgramGroupOptions raygen_group_options = {};
		UtilsBuffer::CUDABuffer RayGenBuffer;

	};

	class missShaders {
	public:missShaders() {}
		  missShaders(const std::vector<OptixModule>& modules, const std::vector<std::string>& funcNames, OptixDeviceContext context)
		  {
			  miss_prog_groups.resize(modules.size());
			  miss_prog_group_descs.resize(modules.size());
			  unsigned int count = 0;
			  for (auto name : funcNames)
			  {
				  miss_prog_group_descs[count].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				  miss_prog_group_descs[count].miss.module = modules[count];
				  miss_prog_group_descs[count].miss.entryFunctionName = funcNames[count].c_str();
				  count++;
			  }
			  OPTIX_CHECK_LOG(optixProgramGroupCreate(
				  context,
				  miss_prog_group_descs.data(),
				  miss_prog_group_descs.size(),   // num program groups
				  &miss_group_option,
				  LOG, &LOG_SIZE,
				  miss_prog_groups.data()
			  ));
		  }
		  void load(const std::vector<OptixModule>& modules, const std::vector<std::string>& funcNames, OptixDeviceContext context)
		  {
			  miss_prog_groups.resize(modules.size());
			  miss_prog_group_descs.resize(modules.size());
			  unsigned int count = 0;
			  for (auto name : funcNames)
			  {
				  miss_prog_group_descs[count].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				  miss_prog_group_descs[count].miss.module = modules[count];
				  miss_prog_group_descs[count].miss.entryFunctionName = funcNames[count].c_str();
				  count++;
			  }
			  OPTIX_CHECK_LOG(optixProgramGroupCreate(
				  context,
				  miss_prog_group_descs.data(),
				  miss_prog_group_descs.size(),   // num program groups
				  &miss_group_option,
				  LOG, &LOG_SIZE,
				  miss_prog_groups.data()
			  ));
		  }
		  template<class T>
		  void setSbtRecord(std::vector<T>& SbtRecord, OptixShaderBindingTable& Sbt) {
			  for (unsigned int i=0;i<SbtRecord.size();i++)
			  {
				  OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups[i], &SbtRecord[i]));
			  }
			  MissBuffer.alloc_and_upload(SbtRecord);
			  Sbt.missRecordBase = MissBuffer.d_pointer();
			  Sbt.missRecordStrideInBytes = sizeof(T);
			  Sbt.missRecordCount = (unsigned int)SbtRecord.size();
		  }
		  ~missShaders() { MissBuffer.free(); }
		std::vector<OptixProgramGroup> miss_prog_groups = {};
		std::vector<OptixProgramGroupDesc> miss_prog_group_descs = {};
		OptixProgramGroupOptions miss_group_option = {};
		UtilsBuffer::CUDABuffer MissBuffer;
	};

	struct hitModulesData {
		hitModulesData(OptixModule is, OptixModule  ah, OptixModule ch):isModule(is), ahModule(ah), chModule(ch) {}
		hitModulesData() {}
		OptixModule isModule=nullptr;
		OptixModule ahModule=nullptr;
		OptixModule chModule=nullptr;
	};
	struct hitFuncNamesData {
		hitFuncNamesData(std::string is, std::string ah, std::string ch) :isName(is),ahName(ah),chName(ch) {}
		hitFuncNamesData() {}
		std::string isName = "null";
		std::string ahName = "null";
		std::string chName = "null";
	};

	class hitShaders {
	public:hitShaders() {}
		  hitShaders(const std::vector<hitModulesData>& modules, const std::vector<hitFuncNamesData>& funcNames, OptixDeviceContext context)
		  {
			  hit_prog_groups.resize(modules.size());
			  hit_prog_group_descs.resize(modules.size());
			  unsigned int count = 0;
			  for (auto& hitModule : modules)
			  {
				  hit_prog_group_descs[count].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				  for (unsigned int i = 0; i < 3; i++)
				  {
					  if (i == IS_INDEX && hitModule.isModule != nullptr) {
						  hit_prog_group_descs[count].hitgroup.moduleIS = modules[count].isModule;
						  hit_prog_group_descs[count].hitgroup.entryFunctionNameIS=funcNames[count].isName.c_str();
					  }
					  if (i == AH_INDEX && hitModule.ahModule != nullptr) {
						  hit_prog_group_descs[count].hitgroup.moduleAH = modules[count].ahModule;
						  hit_prog_group_descs[count].hitgroup.entryFunctionNameAH = funcNames[count].ahName.c_str();
					  }
					  if (i == CH_INDEX && hitModule.chModule != nullptr) {
						  hit_prog_group_descs[count].hitgroup.moduleCH = modules[count].chModule;
						  hit_prog_group_descs[count].hitgroup.entryFunctionNameCH = funcNames[count].chName.c_str();
					  }
				  }
				  count++;
			  }

			  OPTIX_CHECK_LOG(optixProgramGroupCreate(
				  context,
				  hit_prog_group_descs.data(),
				  hit_prog_group_descs.size(),   // num program groups
				  &hit_group_option,
				  LOG, &LOG_SIZE,
				  hit_prog_groups.data()
			  ));
		  }
		  void load(const std::vector<hitModulesData>& modules, const std::vector<hitFuncNamesData>& funcNames, OptixDeviceContext context)
		  {
			  hit_prog_groups.resize(modules.size());
			  hit_prog_group_descs.resize(modules.size());
			  unsigned int count = 0;
			  for (auto& hitModule : modules)
			  {
				  hit_prog_group_descs[count].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				  for (unsigned int i = 0; i < 3; i++)
				  {
					  if (i == IS_INDEX && hitModule.isModule != nullptr) {
						  hit_prog_group_descs[count].hitgroup.moduleIS = modules[count].isModule;
						  hit_prog_group_descs[count].hitgroup.entryFunctionNameIS = funcNames[count].isName.c_str();
					  }
					  if (i == AH_INDEX && hitModule.ahModule != nullptr) {
						  hit_prog_group_descs[count].hitgroup.moduleAH = modules[count].ahModule;
						  hit_prog_group_descs[count].hitgroup.entryFunctionNameAH = funcNames[count].ahName.c_str();
					  }
					  if (i == CH_INDEX && hitModule.chModule != nullptr) {
						  hit_prog_group_descs[count].hitgroup.moduleCH = modules[count].chModule;
						  hit_prog_group_descs[count].hitgroup.entryFunctionNameCH = funcNames[count].chName.c_str();
					  }
				  }
				  count++;
			  }

			  OPTIX_CHECK_LOG(optixProgramGroupCreate(
				  context,
				  hit_prog_group_descs.data(),
				  hit_prog_group_descs.size(),   // num program groups
				  &hit_group_option,
				  LOG, &LOG_SIZE,
				  hit_prog_groups.data()
			  ));
		  }
		  template<class T,unsigned int RAY_COUNT>
		  void setSbtRecord(std::vector<T>& SbtRecord, OptixShaderBindingTable& Sbt) {
			  unsigned int hit_count = 0;
			  for (size_t SBT_COUNT = 0; SBT_COUNT < SbtRecord.size(); SBT_COUNT++)
			  {
				  if (hit_count == RAY_COUNT)
				  {
					  //������hit_count������RAY_COUNT��ʱ������Ϊ0��Ϊ�˱�֤SbtRecord��
					  //ÿ����Optix�ύSBT��ʱ���ǰ�ÿ��ͼԪ��SBT�ֱ������еĹ����¼��ύ���ٽ�����һ��ͼԪ��SBT�ύ
					  //����Ҫע��װ��SBTRecord��ʱ��ҲҪ�����˳��װ��
					  hit_count = 0;
				  }
				  OPTIX_CHECK(optixSbtRecordPackHeader(hit_prog_groups[hit_count], &SbtRecord[SBT_COUNT]));
				  hit_count++;
			  }
			  HitBuffer.alloc_and_upload(SbtRecord);
			  Sbt.hitgroupRecordBase = HitBuffer.d_pointer();
			  Sbt.hitgroupRecordStrideInBytes = sizeof(T);
			  Sbt.hitgroupRecordCount = (unsigned int)(SbtRecord.size());
		  }

		  ~hitShaders() { HitBuffer.free(); }
	
		std::vector<OptixProgramGroup> hit_prog_groups = {};
		std::vector<OptixProgramGroupDesc> hit_prog_group_descs = {};
		OptixProgramGroupOptions hit_group_option = {}; 
		UtilsBuffer::CUDABuffer HitBuffer;
	};

	//class callShaders {
	//public:callShaders() {}
	//	  callShaders(const std::vector<OptixModule[2]>& modules, const std::vector<std::string[2]>& funcNames, OptixDeviceContext context)
	//	  {
	//		  call_prog_groups.resize(modules.size());
	//		  call_prog_group_descs.resize(modules.size());
	//		  unsigned int count = 0;
	//		  for (auto name : funcNames)
	//		  {
	//			  call_prog_group_descs[count].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
	//			  for (unsigned int i = 0; i < 2; i++)
	//			  {
	//				  if (i == DC_INDEX && modules[count][DC_INDEX] != nullptr) {
	//					  call_prog_group_descs[count].callables.moduleDC = modules[count][DC_INDEX];
	//					  call_prog_group_descs[count].callables.entryFunctionNameDC = funcNames[count][DC_INDEX].c_str();
	//				  }
	//				  if (i == CC_INDEX && modules[count][CC_INDEX] != nullptr) {
	//					  call_prog_group_descs[count].callables.moduleDC = modules[count][CC_INDEX];
	//					  call_prog_group_descs[count].callables.entryFunctionNameDC = funcNames[count][CC_INDEX].c_str();
	//				  }
	//			  }
	//			  count++;
	//		  }

	//		  OPTIX_CHECK_LOG(optixProgramGroupCreate(
	//			  context,
	//			  call_prog_group_descs.data(),
	//			  call_prog_group_descs.size(),   // num program groups
	//			  &call_group_option,
	//			  LOG, &LOG_SIZE,
	//			  call_prog_groups.data()
	//		  ));
	//	  }
	//private:
	//	std::vector<OptixProgramGroup> call_prog_groups = {};
	//	std::vector<OptixProgramGroupDesc> call_prog_group_descs = {};
	//	OptixProgramGroupOptions call_group_option = {};
	//	UtilsBuffer::CUDABuffer CallBuffer;
	//};

	//�������е�shader
	class Shaders {
	public:
		Shaders() {}
		Shaders(std::string ShaderSoureFilePath) { Params::getPTXCode(ptx, ShaderSoureFilePath); }
		void setShader(std::string JSONFilePath,OptixModule module,OptixDeviceContext context) {
			// ��ȡjson�ļ�(ע��json�����ǲ���BOM��UTF-8����)
			std::ifstream ifs(JSONFilePath, std::ios::binary);
			if (!ifs.is_open())
			{
				std::cerr << "Failed to open json file!" << std::endl;
			}
			std::string content((std::istreambuf_iterator<char>(ifs)),
				(std::istreambuf_iterator<char>()));
			ifs.close();
			std::cout << "Json content: " << content << std::endl;

			// ����json
			rapidjson::Document doc;
			if (doc.Parse(content.c_str()).HasParseError())
			{
				std::cerr << "Failed to parse json file!" << std::endl;
				std::cerr << "Error code: " << doc.GetParseError() << std::endl;
				std::cerr << "Error offset: " << doc.GetErrorOffset() << std::endl;
			}

			// �����ڵ��Ƿ�ΪObject
			if (!doc.IsObject())
			{
				std::cerr << "Invalid uniform_layout.json format! Root should be an Object." << std::endl;
			}
			/*����rayGenshader*/
			OptixModule rayGenModule = nullptr;
			std::string rayGenFuncName;
			// ��ȡrayGenshader����
			const auto& RayGenShader = doc["RayGenShader"];
			const std::string RayGenModuleName = RayGenShader["ModuleName"].GetString();
			rayGenModule=module;
			rayGenFuncName = RayGenShader["FuncName"].GetString();
			rayGenShader.load(rayGenModule, rayGenFuncName,context);

			/*����missShader*/
			std::vector<OptixModule>missModules = {};
			std::vector<std::string>missModulesFuncNames = {};
			// ��ȡmissShader����
			const auto& MissShader = doc["MissShader"];
			const auto& MissData = MissShader["Data"];
			for (uint16_t i = 0; i < MissData.Size(); i++) {
				OptixModule tempmodule = nullptr;
				const auto& obj = MissData[i];
				// ��ȡ�����ModuleName����
				const std::string ModuleName = obj["ModuleName"].GetString();
				// ��ȡ�����FuncName����
				const std::string FuncName = obj["FuncName"].GetString();
				tempmodule = module;
				missModules.push_back(tempmodule);
				missModulesFuncNames.push_back(FuncName);
			}
			missShader.load(missModules, missModulesFuncNames,context);

			/*����hitshader*/
			std::vector<hitModulesData> hitModules = {};
			std::vector<hitFuncNamesData> hitModulesFuncNames = {};
			// ��ȡhitShader����
			const auto& HitShader = doc["HitShader"];
			//��ȡhitShader��ModuleName����
			const auto& HitData = HitShader["Data"];
			// ����Data�����е�ÿ������
			for (uint16_t i = 0; i < HitData.Size(); i++)
			{
				hitModulesData moduleData = {};
				const auto& obj = HitData[i];
				// ��ȡ�����ISModuleName����
				const std::string ISModuleName = obj["ISModuleName"].GetString();
				// ��ȡ�����AHModuleName����
				const std::string AHModuleName = obj["AHModuleName"].GetString();
				// ��ȡ�����CHModuleName����
				const std::string CHModuleName = obj["CHModuleName"].GetString();
				if (ISModuleName == "NULL")
					moduleData.isModule = nullptr;
				else
					moduleData.isModule = module;
				if (AHModuleName == "NULL")
					moduleData.ahModule = nullptr;
				else
					moduleData.ahModule = module;
				if (CHModuleName == "NULL")
					moduleData.chModule = nullptr;
				else
					moduleData.chModule = module;

				hitModules.push_back(moduleData);

				hitFuncNamesData FuncData = {};
				// ��ȡ�����ISModuleName����
				const std::string ISName = obj["ISName"].GetString();
				// ��ȡ�����AHModuleName����
				const std::string AHName = obj["AHName"].GetString();
				// ��ȡ�����CHModuleName����
				const std::string CHName = obj["CHName"].GetString();

				FuncData.isName = ISName;
				FuncData.ahName = AHName;
				FuncData.chName = CHName;

				hitModulesFuncNames.push_back(FuncData);
			}
			hitShader.load(hitModules, hitModulesFuncNames,context);
		}
		std::string* getPTX() { return &ptx; }

	public:
		rayGenShaders rayGenShader = {};
		missShaders missShader = {};
		hitShaders hitShader = {};
	private:
		std::string ptx;
	};
}