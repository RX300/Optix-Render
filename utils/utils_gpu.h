#pragma once
#include"utils.h"
float4 readCudaTex4(cudaTextureObject_t texture, float u, float v)
{
	return tex2D<float4>(texture, u, v);
}