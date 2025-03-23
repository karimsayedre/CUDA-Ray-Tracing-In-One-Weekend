#pragma once
#include "pch.cuh"

// Host code to set up the surface object
inline cudaSurfaceObject_t SetupFramebufferSurface(uint32_t width, uint32_t height)
{
	// Allocate CUDA array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaArray_t			  cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);

	// Create surface object
	cudaResourceDesc resDesc;
	resDesc.resType			= cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaSurfaceObject_t surfObj;
	cudaCreateSurfaceObject(&surfObj, &resDesc);

	return surfObj;
}