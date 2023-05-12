#pragma once
#include "shaders.h"
#include "utils/Params.h"
#include "rapidjson/document.h"
#include <optix_stack_size.h>
#include <iomanip>
#include <set>
namespace PipLine
{
	static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
	{
		std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
				  << message << "\n";
	}
	class pipline
	{
	public:
		pipline(std::string LaunchParamsVariableName, unsigned int numPayloadValues, unsigned int numAttributeValues)
		{
			/*初始化设备*/
			// Initialize CUDA
			CUDA_CHECK(cudaFree(0));

			// Initialize the OptiX API, loading all API entry points
			OPTIX_CHECK(optixInit());

			// Specify context options
			OptixDeviceContextOptions options = {};
			options.logCallbackFunction = &context_log_cb;
			options.logCallbackLevel = 4;

			// Associate a CUDA context (and therefore a specific GPU) with this
			// device context
			CUcontext cuCtx = 0; // zero means take the current context
			OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

			/*初始化管线编译设置并将cu文件编译成模块*/
			{
				PipLineLaunchParamsVariableName = LaunchParamsVariableName;
#if !defined(NDEBUG)
				module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
				module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
				pipeline_compile_options.usesMotionBlur = false;
				pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
				pipeline_compile_options.numPayloadValues = numPayloadValues;
				pipeline_compile_options.numAttributeValues = numAttributeValues;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
				pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
				pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
				pipeline_compile_options.pipelineLaunchParamsVariableName = PipLineLaunchParamsVariableName.c_str();
				pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
			}
		}
		void loadshader(Shaders::Shaders &shader)
		{
			std::string *inputPTX = shader.getPTX();
			OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
				context,
				&module_compile_options,
				&pipeline_compile_options,
				inputPTX->c_str(),
				inputPTX->size(),
				LOG, &LOG_SIZE,
				&module));
		}

		void LinkShader(Shaders::Shaders &shader)
		{
			std::string *inputPTX = shader.getPTX();
			OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
				context,
				&module_compile_options,
				&pipeline_compile_options,
				inputPTX->c_str(),
				inputPTX->size(),
				LOG, &LOG_SIZE,
				&module));
			/*add rayGenShader*/
			addShader(shader.rayGenShader.raygen_prog_group);
			/*add missShader*/
			addShaders(shader.missShader.miss_prog_groups);
			/*add hitShader*/
			addShaders(shader.hitShader.hit_prog_groups);
		}

		void LinkPipeline(uint32_t pipline_max_trace)
		{
			max_trace_depth = pipline_max_trace;
			pipeline_link_options.maxTraceDepth = max_trace_depth;
			pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

			OPTIX_CHECK_LOG(optixPipelineCreate(
				context,
				&pipeline_compile_options,
				&pipeline_link_options,
				programGroups.data(),
				programGroups.size(),
				LOG, &LOG_SIZE,
				&pipeline));

			OptixStackSizes stack_sizes = {};
			for (auto &prog_group : programGroups)
			{
				OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
			}

			uint32_t direct_callable_stack_size_from_traversal;
			uint32_t direct_callable_stack_size_from_state;
			uint32_t continuation_stack_size;
			OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
												   0, // maxCCDepth
												   0, // maxDCDEpth
												   &direct_callable_stack_size_from_traversal,
												   &direct_callable_stack_size_from_state, &continuation_stack_size));
			OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
												  direct_callable_stack_size_from_state, continuation_stack_size,
												  1 // maxTraversableDepth
												  ));
		}

		OptixDeviceContext getDevice() { return this->context; }
		OptixPipeline getPipLine() { return this->pipeline; }
		OptixModule getModule() { return this->module; }

		~pipline()
		{
			OPTIX_CHECK(optixPipelineDestroy(pipeline));
			OPTIX_CHECK(optixModuleDestroy(module));
			OPTIX_CHECK(optixDeviceContextDestroy(context));
		}

	private:
		void addShader(const OptixProgramGroup &pg)
		{
			programGroups.push_back(pg);
		}
		void addShaders(const std::vector<OptixProgramGroup> &pgs)
		{
			for (auto &pg : pgs)
				programGroups.push_back(pg);
		}

	public:
		OptixDeviceContext context = nullptr;
		uint32_t max_trace_depth = 0;
		std::string PipLineLaunchParamsVariableName; // 整条管线上的自定义数据，用于用户和optix端通信
	private:
		OptixPipeline pipeline = nullptr;
		OptixPipelineCompileOptions pipeline_compile_options = {};
		OptixModuleCompileOptions module_compile_options = {};
		OptixModule module = nullptr;
		std::vector<OptixProgramGroup> programGroups = {};
		OptixPipelineLinkOptions pipeline_link_options = {};
		OptixStackSizes stack_sizes = {};
	};
}