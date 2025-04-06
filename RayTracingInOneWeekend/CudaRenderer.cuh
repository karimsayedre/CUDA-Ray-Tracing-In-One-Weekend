#pragma once
#include "pch.cuh"
#include <SFML/Graphics/Image.hpp>
#include "Camera.cuh"

#if 1
__host__ void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line);
#define CHECK_CUDA_ERRORS(val) CheckCuda((val), #val, __FILE__, __LINE__)
#else
#define CHECK_CUDA_ERRORS(val) (val)
#endif

namespace BVH
{
	struct BVHSoA;
}

namespace Hitables
{
	struct HittableList;
}

namespace Mat
{
	struct Materials;
}

struct Dimensions
{
	uint32_t Width;
	uint32_t Height;
};

struct HitRecord
{
	Vec3	 Location;
	Vec3	 Normal;
	uint16_t PrimitiveIndex;
};

class CudaRenderer
{
  public:
	__host__ CudaRenderer(const Dimensions dims, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul);

	__host__ void ResizeImage(const uint32_t width, const uint32_t height);

	__host__ std::chrono::duration<float, std::milli> Render(const uint32_t width, const uint32_t height) const;

	__host__ ~CudaRenderer();

	cudaArray_const_t GetImageArray() const
	{
		return d_ImageArray;
	}

  private: // device pointers
	Hitables::HittableList* d_List {};
	BVH::BVHSoA*			d_World {};
	Mat::Materials*			d_Materials {};
	Camera*					d_Camera {};

	uint32_t*			d_RandSeeds {};
	cudaSurfaceObject_t d_Image {};
	cudaArray_t			d_ImageArray {};

  private: // host variables
	uint32_t	   m_SamplesPerPixel;
	uint32_t	   m_MaxDepth;
	float		   m_ColorMul;
	mutable Camera m_Camera;
};
