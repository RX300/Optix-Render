
#define NOMINMAX
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <nanovdb/util/CudaDeviceBuffer.h> // required for cuda memory management

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Trackball.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

#include "geometry/assimp_model.h"
#include "pipline/shaders.h"
#include "material/material_definition.h"
#include <sutil/Camera.h>
#include "geometry/volume_vdb.h"
#include "pipline/PipLine.h"

#include <memory>
#include <format>
#include <vector>
#include <chrono>

void configureCamera(sutil::Camera &cam, const uint32_t width, const uint32_t height)
{
    // cam.setEye({ 0.0f, 0.5f, 14.5f });
    // cam.setLookat({ 0.0f, 0.5f, 0.0f });
    ////cam.setUp( {0.0f, 1.0f, 3.0f} );
    // cam.setUp({ 0.0f, 1.0f, 0.0f });
    cam.setEye({1.8, -4.0, 1.8f});
    cam.setLookat({1.8, .0, 1.8f});
    // cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setUp({0.0, 0.0f, 1.0f});
    cam.setFovY(45.0f);
    cam.setAspectRatio((float)width / (float)height);
}

void printUsageAndExit(const char *argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit(1);
}

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

int main(int argc, char *argv[])
{
    // 计算初始化时间，设立开始的时间点
    auto start = std::chrono::high_resolution_clock::now();

    std::string outfile;
    int width = 1000;
    int height = 1000;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i < argc - 1)
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), width, height);
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        //
        // Initialize CUDA and create OptiX context
        //
        PipLine::pipline pipline("params", 5, 3);
        auto context = pipline.getDevice();
        /*Gemotery::assimp_model model("assets/cornell_box_Marry.obj");
        model.setLight(4);*/
        Gemotery::volume_vdb model("../../assets/CTA_test_NANO.nvdb");
        OptixGas::gasBuild gas;
        gas.addGasHit(&model);
        OptixTraversableHandle gas_handle = gas.buildGAS(context);
        // model.setLight(4);
        // OptixTraversableHandle gas_handle = model.buildGAS(context);

        /* Create shaders*/
        Shaders::Shaders shader("../../volume.optixir");
        pipline.loadshader(shader);
        auto module = pipline.getModule();
        shader.setShader("../../Shaders.json", module, context);

        //
        // Link pipeline
        //
        {
            const uint32_t max_trace_depth = 4;
            pipline.LinkShader(shader);
            pipline.LinkPipeline(max_trace_depth);
        }
        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            // rayGen record
            std::vector<RayGenSbtRecord> rayRecord;
            RayGenSbtRecord raysbt;
            raysbt.data = {};
            rayRecord.push_back(raysbt);
            shader.rayGenShader.setSbtRecord<RayGenSbtRecord>(rayRecord, sbt);
            // miss record
            std::vector<MissSbtRecord> missRecords;
            for (int i = 0; i < 1; i++)
            {
                MissSbtRecord tempRecord;
                tempRecord.data = {0.2f, 0.2f, 0.2f};
                missRecords.push_back(tempRecord);
            }
            shader.missShader.setSbtRecord<MissSbtRecord>(missRecords, sbt);
            // hit record
            /* std::vector<DisneySbtRecord>hitRecords;
             for (auto& iter : model.meshes)
             {
                 for (int i = 0; i < 1; i++)
                 {
                     DisneySbtRecord tempSbtRecord;
                     tempSbtRecord.data = iter->matInfo;
                     hitRecords.push_back(tempSbtRecord);
                 }
             }*/

            std::vector<VolumeSbtRecord> hitRecords;
            model.drawHit(shader.hitShader, hitRecords, 1);
            shader.hitShader.setSbtRecord<VolumeSbtRecord, 1>(hitRecords, sbt);
        }

        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
        // 计算初始化时间，设立结束的时间点
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "init time: " << elapsed.count() << "s" << std::endl;
        // 计算渲染时间，设立开始的时间点
        start = std::chrono::high_resolution_clock::now();

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            sutil::Camera cam;
            configureCamera(cam, width, height);

            Params::Params params;
            params.image = output_buffer.map();
            params.image_width = width;
            params.image_height = height;
            params.handle = gas_handle;
            params.cam_eye = cam.eye();
            cudaMalloc((void **)&params.states, 2 * width * height * sizeof(curandState));
            // initOneState(params.states);
            cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);

            CUdeviceptr d_param;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params::Params)));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(d_param),
                &params, sizeof(params),
                cudaMemcpyHostToDevice));
            auto optix_pipline = pipline.getPipLine();
            OPTIX_CHECK(optixLaunch(optix_pipline, stream, d_param, sizeof(Params::Params), &sbt, width, height, /*depth=*/1));
            CUDA_SYNC_CHECK();

            output_buffer.unmap();
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
        }
        // 计算渲染时间，设立结束的时间点
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "render time: " << elapsed.count() << "s" << std::endl;
        //
        // Display results
        //
        {
            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = width;
            buffer.height = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            if (outfile.empty())
                sutil::displayBufferWindow(argv[0], buffer);
            else
                sutil::saveImage(outfile.c_str(), buffer, false);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
