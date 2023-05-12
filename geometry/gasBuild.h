#pragma once
#include <vector>
#include <format>
#include <string>
#include <optix.h>
#include <sutil/Exception.h>
#include "../buffer/CUDABuffer.h"
#include "../pipline/shaders.h"
namespace OptixGas
{

	class gasHitObject
	{
	public:
		gasHitObject() {}
		// 向GAS(几何加速结构)里存数据，需要传递一个OptixBuildInput的vector作为参数，会让gasBuild类里存
		// 一个该vector类，然后用这个vector取buildGAS
		virtual void inputGasData(std::vector<OptixBuildInput> &) = 0;
		// 析构函数，用于销毁gpu上的数据
		virtual ~gasHitObject() {}
	};

	class gasBuild
	{
	public:
		gasBuild() {}
		// build GAS
		OptixTraversableHandle buildGAS(const OptixDeviceContext &optixContext);
		// 加入要被build的hitObject
		bool addGasHit(gasHitObject *hit);

	private:
		// 存放要向optix提交的数据
		std::vector<OptixBuildInput> buildData;
		// 为减少复制开销直接存gasHitTable的指针
		std::vector<gasHitObject *> hits;
		// 存放as(加速结构)的buffer
		UtilsBuffer::CUDABuffer asBuffer;
	};
	inline OptixTraversableHandle gasBuild::buildGAS(const OptixDeviceContext &optixContext)
	{
		for (auto i : hits)
			i->inputGasData(this->buildData);

		// ==================================================================
		// BLAS setup
		// ==================================================================
		
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		try
		{
			OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
													 &accelOptions,
													 buildData.data(),
													 buildData.size(), // num_build_inputs
													 &blasBufferSizes));
		}
		catch (std::exception &e)
		{
			std::cerr << "Caught exception: " << e.what() << "\n";
		}
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

		OptixTraversableHandle asHandle{0};
		OPTIX_CHECK(optixAccelBuild(optixContext,
									/* stream */ 0,
									&accelOptions,
									buildData.data(),
									buildData.size(),
									tempBuffer.d_pointer(),
									tempBuffer.sizeInBytes,

									outputBuffer.d_pointer(),
									outputBuffer.sizeInBytes,

									&asHandle,

									&emitDesc, 1));
		CUDA_SYNC_CHECK();
		// ==================================================================
		// perform compaction
		// ==================================================================
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);

		this->asBuffer.alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixContext,
									  /*stream:*/ 0,
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
	inline bool gasBuild::addGasHit(gasHitObject *hit)
	{
		if (find(hits.begin(), hits.end(), hit) != hits.end())
		{
			// hit already exists in the vector
			std::cout << "this element has already exists" << std::endl;
			return false;
		}
		else
		{
			// hit does not exist in the vector, add it
			hits.push_back(hit);
			return true;
		}
	}
}