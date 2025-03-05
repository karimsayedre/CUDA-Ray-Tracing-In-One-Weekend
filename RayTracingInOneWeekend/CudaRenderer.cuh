#pragma once
#include "pch.cuh"
#include <glm/glm.hpp>
#include <corecrt_math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <SFML/Graphics/Image.hpp>

#include "Hittable.h"
#include "Material.h"

#define CHECK_CUDA_ERRORS(val) check_cuda((val), #val, __FILE__, __LINE__)

namespace std
{
	class mutex;
}

class Camera;
__host__ void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

//__device__ vec3 unit_vector(const vec3& v);

__device__ glm::vec3 color(const Ray& r);

class CudaRenderer
{
  public:
	__host__ CudaRenderer(const uint32_t width, const uint32_t height, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul)
		: m_Width(width), m_Height(height), m_SamplesPerPixel(samplesPerPixel), m_MaxDepth(maxDepth), m_ColorMul(colorMul)
	{
		CHECK_CUDA_ERRORS(cudaMallocManaged(&d_Image, m_Width * m_Height * sizeof(float3)));
	}

	__host__ void Init();

	__host__ void Render() const;
	__host__ std::vector<float> CopyImage();

  private:
	Hittable*  d_list;
	Material*  d_materials;
	Hittable** d_world;
	Camera*	   d_camera;

	uint32_t*	 d_rand_seeds;
	curandState* d_rand_state2;

  private:
	uint32_t   m_Width;
	uint32_t   m_Height;
	uint32_t   m_SamplesPerPixel;
	uint32_t   m_MaxDepth;
	float	   m_ColorMul;
	glm::vec3* d_Image;
};
