#pragma once
#include "pch.cuh"
#include <glm/glm.hpp>
#include <corecrt_math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <SFML/Graphics/Image.hpp>

#include "Camera.cuh"

#if 1
#define CHECK_CUDA_ERRORS(val) CheckCuda((val), #val, __FILE__, __LINE__)
#else
#define CHECK_CUDA_ERRORS(val) (val)
#endif

struct HittableList;
struct Materials;
struct BVHSoA;

struct HitRecord
{
	Vec3	 Location;
	Vec3	 Normal;
	float	 T;
	uint16_t MaterialIndex;
};

class Camera;
__host__ void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line);

class CudaRenderer
{
  public:
	__host__ CudaRenderer(const uint32_t width, const uint32_t height, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul);

	__host__ void Render() const;

	__host__ ~CudaRenderer();

	__host__ auto GetDeviceImage() const
	{
		return d_Image;
	}

  private: // device pointers
	HittableList* d_List {};
	BVHSoA*		  d_World {};
	Materials*	  d_Materials {};
	Camera*		  d_Camera {};

	uint32_t*			d_RandSeeds {};
	cudaSurfaceObject_t d_Image {};

  private: // host variables
	uint32_t	   m_Width;
	uint32_t	   m_Height;
	uint32_t	   m_SamplesPerPixel;
	uint32_t	   m_MaxDepth;
	float		   m_ColorMul;
	mutable Camera m_Camera;
};
