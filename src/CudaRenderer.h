#pragma once
#include "pch.h"
#include "Camera.h"
#include "SFML/System/Vector2.hpp"

#define CHECK_CUDA
#ifdef CHECK_CUDA
__host__ void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line);
#define CHECK_CUDA_ERRORS(val) CheckCuda((val), #val, __FILE__, __LINE__)
#else
#define CHECK_CUDA_ERRORS(val) (val)
#endif

#define CHECK_BOOL(val)     \
	do                      \
	{                       \
		if (!(val))         \
		{                   \
			__debugbreak(); \
			assert(false);  \
		}                   \
	} while (false)

namespace BVH
{
	struct BVHSoA;
}

namespace Hitables
{
	struct PrimitiveList;
}

namespace Mat
{
	struct Materials;
}

struct alignas(64) HitRecord
{
	Vec3	 Location;
	Vec3	 Normal;
	uint16_t PrimitiveIndex;
};

struct RenderParams
{
	Hitables::PrimitiveList* __restrict__ List {};
	BVH::BVHSoA* __restrict__ BVH {};
	Mat::Materials* __restrict__ Materials {};
	uint32_t* __restrict__ RandSeeds {};
	CameraPOD			Camera {};
	float4				ResolutionInfo {};
	cudaSurfaceObject_t Image {};
	uint32_t			m_SamplesPerPixel {};
	uint32_t			m_MaxDepth {};
	float				m_ColorMul {};
};

__constant__ inline RenderParams d_Params;

class CudaRenderer
{
  public:
	__host__ CudaRenderer(const sf::Vector2u dims, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul);

	__host__ void ReleaseResizables() const;
	void		  ResizeImage(const sf::Vector2u dims, cudaSurfaceObject_t surface);

	__host__ void							 CopyDeviceData(const cudaSurfaceObject_t imageSurface);
	std::chrono::duration<float, std::milli> Render(const sf::Vector2u& size, cudaSurfaceObject_t surface);

	__host__ ~CudaRenderer();

  private: // device pointers
	Hitables::PrimitiveList* dp_List {};
	BVH::BVHSoA*			 dp_BVH {};
	Mat::Materials*			 dp_Materials {};
	uint32_t*				 dp_RandSeeds {};

  private: // host variables
	uint32_t	   m_SamplesPerPixel;
	uint32_t	   m_MaxDepth;
	float		   m_ColorMul;
	mutable Camera m_Camera;

	sf::Vector2u m_Dims {};

	cudaEvent_t m_StartEvent, m_EndEvent;

	RenderParams h_Params;

  public:
	cudaGraphicsResource* cudaResource = nullptr;
};
