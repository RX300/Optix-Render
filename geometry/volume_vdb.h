#pragma once
#include <array>
#include "buffer/CUDABuffer.h"
#include "gasBuild.h"
#include <nanovdb/util/IO.h>               // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/util/CudaDeviceBuffer.h> // required for CUDA memory management
#include "material/material_definition.h"
#include <sutil/Aabb.h>
namespace Gemotery
{
    class volume_vdb : public OptixGas::gasHitObject
    {
    public:
        // 构造函数，参数是vdb文件路径
        volume_vdb(const std::string &pathname);
        // 重写父类的虚函数
        virtual void inputGasData(std::vector<OptixBuildInput> &) override;
        virtual ~volume_vdb() override
        {
            delete v_mat;
            d_verticesBuffer.free();
            d_indicesBuffer.free();
        }
        void drawHit(Shaders::hitShaders &hitShader, std::vector<VolumeSbtRecord> &hitRecords, unsigned int RAY_COUNT);
        Material::volume_mat *get_volume() { return this->v_mat; }

    private:
        // nanovdb文件的栅格数据
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> gridHandle;
        float3 box_min;
        float3 box_max;
        std::vector<float3> box_vertices;
        std::vector<uint3> box_indices;
        CUdeviceptr d_vertice; // 单独设置一个这个是因为triangle_input[count].triangleArray.vertexBuffers
        // 要的是CUDeviceptr的地址，必须要保存为左值
        UtilsBuffer::CUDABuffer d_verticesBuffer;
        UtilsBuffer::CUDABuffer d_indicesBuffer;
        // 材质
        Material::volume_mat *v_mat;
    };

    // 构造函数，参数是vdb文件路径
    inline volume_vdb::volume_vdb(const std::string &pathname)
    {
        // 读取nanovdb文件
        gridHandle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(pathname);
        v_mat = new Material::volume_mat(gridHandle);
        auto *cpugrid = gridHandle.grid<float>();
        // 获取栅格数据的包围盒
        auto temp_min = cpugrid->worldBBox().min();
        auto temp_max = cpugrid->worldBBox().max();
        box_min = make_float3(temp_min[0], temp_min[1], temp_min[2]);
        box_max = make_float3(temp_max[0], temp_max[1], temp_max[2]);
        // 打印包围盒
        std::cout << "box_min:" << box_min.x << "," << box_min.y << "," << box_min.z << std::endl;
        std::cout << "box_max:" << box_max.x << "," << box_max.y << "," << box_max.z << std::endl;
        // 获取包围盒的8个顶点
        box_vertices.resize(8);
        box_vertices[0] = make_float3(box_max.x, box_min.y, box_min.z);
        box_vertices[1] = make_float3(box_max.x, box_max.y, box_min.z);
        box_vertices[2] = make_float3(box_max.x, box_max.y, box_max.z);
        box_vertices[3] = make_float3(box_max.x, box_min.y, box_max.z);
        box_vertices[4] = make_float3(box_min.x, box_min.y, box_min.z);
        box_vertices[5] = make_float3(box_min.x, box_max.y, box_min.z);
        box_vertices[6] = make_float3(box_min.x, box_max.y, box_max.z);
        box_vertices[7] = make_float3(box_min.x, box_min.y, box_max.z);
        // 打印包围盒的8个顶点
        for (int i = 0; i < 8; i++)
        {
            std::cout << "box_vertices[" << i << "]:" << box_vertices[i].x << "," << box_vertices[i].y << "," << box_vertices[i].z << std::endl;
        }
        // 获取包围盒的12个面,并填充索引
        box_indices.resize(12);
        //+X
        box_indices[0] = make_uint3(0, 1, 2);
        box_indices[1] = make_uint3(0, 2, 3);
        //-X
        box_indices[2] = make_uint3(4, 7, 6);
        box_indices[3] = make_uint3(4, 6, 5);
        //+Y
        box_indices[4] = make_uint3(1, 5, 6);
        box_indices[5] = make_uint3(1, 6, 2);
        //-Y
        box_indices[6] = make_uint3(0, 3, 7);
        box_indices[7] = make_uint3(0, 7, 4);
        //+Z
        box_indices[8] = make_uint3(3, 2, 6);
        box_indices[9] = make_uint3(3, 6, 7);
        //-Z
        box_indices[10] = make_uint3(0, 4, 5);
        box_indices[11] = make_uint3(0, 5, 1);
        // 打印包围盒的12个面
        for (int i = 0; i < 12; i++)
        {
            std::cout << "box_indices[" << i << "]:" << box_indices[i].x << "," << box_indices[i].y << "," << box_indices[i].z << std::endl;
        }
        // 将顶点数据和索引数据从cpu传到gpu
        d_verticesBuffer.alloc_and_upload(box_vertices);
        d_vertice = d_verticesBuffer.d_pointer(); // 单独设置一个这个是因为triangle_input[count].triangleArray.vertexBuffers
        // 要的是CUDeviceptr的地址，必须要保存为左值
        d_indicesBuffer.alloc_and_upload(box_indices);
    }
    // 重写父类的虚函数 inputGasData
    inline void volume_vdb::inputGasData(std::vector<OptixBuildInput> &Builds)
    {
        const uint32_t triangle_input_flags[] = {OPTIX_GEOMETRY_FLAG_NONE};
        Builds.push_back({});
        // 获取刚才push_back的元素
        auto &triangle_input = Builds.back();
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = 8;
        triangle_input.triangleArray.vertexBuffers = &d_vertice;
        triangle_input.triangleArray.flags = triangle_input_flags;

        triangle_input.triangleArray.vertexStrideInBytes = 0;
        triangle_input.triangleArray.numSbtRecords = 1;
        triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
        triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.numIndexTriplets = 12;
        triangle_input.triangleArray.indexBuffer = d_indicesBuffer.d_pointer();
        // triangle_input.triangleArray.preTransform = nullptr;
        // triangle_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
        triangle_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        Builds.push_back(triangle_input);
        int count = 0;
        /* for (int i = 0; i < 1; i++) {
        //这里不知道为什么，必须加上for循环才可以通过，否则会报错
             OptixBuildInput curbuild = {};
             curbuild.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
             curbuild.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
             curbuild.triangleArray.numVertices = 8;
             curbuild.triangleArray.vertexBuffers = &d_vertice;
             curbuild.triangleArray.flags = triangle_input_flags;

             curbuild.triangleArray.vertexStrideInBytes = 0;
             curbuild.triangleArray.numSbtRecords = 1;
             curbuild.triangleArray.sbtIndexOffsetBuffer = 0;
             curbuild.triangleArray.sbtIndexOffsetSizeInBytes = 0;
             curbuild.triangleArray.sbtIndexOffsetStrideInBytes = 0;

             curbuild.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
             curbuild.triangleArray.indexBuffer = d_indicesBuffer.d_pointer();
             curbuild.triangleArray.numIndexTriplets = 12;
             curbuild.triangleArray.indexStrideInBytes = sizeof(uint3);
             Builds.push_back(curbuild);
         }*/
    }
    // 重写父类的虚函数 drawHit
    inline void volume_vdb::drawHit(Shaders::hitShaders &hitShader, std::vector<VolumeSbtRecord> &hitRecords, unsigned int RAY_COUNT)
    {
        for (int i = 0; i < RAY_COUNT; i++)
        {
            VolumeSbtRecord tempSbtRecord;
            tempSbtRecord.data = *(this->v_mat);
            hitRecords.push_back(tempSbtRecord);
        }
    }

    // 另一种vdb数据体，按照自定义图元的方式向optix提交
    class vdb_volume : public OptixGas::gasHitObject
    {
    public:
        // 构造函数，参数是vdb文件路径
        vdb_volume(const std::string &pathname);
        // 重写父类的虚函数
        virtual void inputGasData(std::vector<OptixBuildInput> &Builds) override;
        void drawHit(Shaders::hitShaders &hitShader) {}

    private:
        // nanovdb文件的栅格数据
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> gridHandle;
        sutil::Aabb aabb;
        CUdeviceptr d_aabb;
    };
    inline vdb_volume::vdb_volume(const std::string &pathname)
    {
        // 读取nanovdb文件
        gridHandle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(pathname);
        auto *cpugrid = gridHandle.grid<float>();
        // get this grid's aabb
        {
            // indexBBox returns the extrema of the (integer) voxel coordinates.
            // Thus the actual bounds of the space covered by those voxels extends
            // by one unit (or one "voxel size") beyond those maximum indices.

            // auto bbox = cpugrid->indexBBox();
            // nanovdb::Coord boundsMin(bbox.min());
            // nanovdb::Coord boundsMax(bbox.max() + nanovdb::Coord(1)); // extend by one unit

            // float3 min = {
            //     static_cast<float>(boundsMin[0]),
            //     static_cast<float>(boundsMin[1]),
            //     static_cast<float>(boundsMin[2]) };
            // float3 max = {
            //     static_cast<float>(boundsMax[0]),
            //     static_cast<float>(boundsMax[1]),
            //     static_cast<float>(boundsMax[2]) };
            auto bbox = cpugrid->worldBBox();
            float3 min = {
                static_cast<float>(bbox.min()[0]),
                static_cast<float>(bbox.min()[1]),
                static_cast<float>(bbox.min()[2])};
            float3 max = {
                static_cast<float>(bbox.max()[0]),
                static_cast<float>(bbox.max()[1]),
                static_cast<float>(bbox.max()[2])};
            // 打印包围盒的min,max
            std::cout << "box_min:" << min.x << "," << min.y << "," << min.z << std::endl;
            std::cout << "box_max:" << max.x << "," << max.y << "," << max.z << std::endl;

            aabb = sutil::Aabb(min, max);
        }

        // up to device
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabb), sizeof(sutil::Aabb)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_aabb), &aabb,
                              sizeof(sutil::Aabb), cudaMemcpyHostToDevice));
    }
    inline void vdb_volume::inputGasData(std::vector<OptixBuildInput> &Builds)
    {
        const uint32_t aabb_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        Builds.push_back({});
        // 获取刚才push_back的元素
        auto &build_input = Builds.back();
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
        build_input.customPrimitiveArray.flags = &aabb_input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
        build_input.customPrimitiveArray.numPrimitives = 1;
        build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
        build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
        build_input.customPrimitiveArray.primitiveIndexOffset = 0;
    }
}
