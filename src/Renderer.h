﻿#pragma once

#include "pch.h"
#include "Camera.h"
#include <chrono>

#include "AABB.h"
#include "CPU_GPU.h"
#include "SFML/System/Vector2.hpp"

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
	uint32_t PrimitiveIndex;
};

struct RenderParams
{
	Hitables::PrimitiveList* __restrict__ List {};
	BVH::BVHSoA* __restrict__ BVH {};
	Mat::Materials* __restrict__ Materials {};
	uint32_t* __restrict__ RandSeeds {};
	CameraPOD			Camera {};
	float4				ResolutionInfo {}; // x: x Pixel Size, y: y Pixel Size, z: width, w: height
	cudaSurfaceObject_t Image {};
	uint32_t			m_SamplesPerPixel {};
	uint32_t			m_MaxDepth {};
	float				m_ColorMul {};
};

extern RenderParams				 h_Params; // unified host copy -> CPURender.cpp
__constant__ inline RenderParams d_Params; // GPU constant memory copy

[[nodiscard]] __host__ __device__ inline RenderParams* GetParams()
{
#if defined(__CUDA_ARCH__)
	// device‐side compilation only: return the constant symbol
	return (RenderParams*)&d_Params;
#else
	// host‐side: return your managed host copy
	return &h_Params;
#endif
}

constexpr int kFramesInFlight = 3;
struct Frame
{
	cudaEvent_t Start, End;
};

template<ExecutionMode Mode>
class Renderer
{
  public:
	__host__ Renderer(const sf::Vector2u dims, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul);

	void ResizeImage(const sf::Vector2u dims, cudaSurfaceObject_t surface);

	__host__ void CopyDeviceData(const cudaSurfaceObject_t imageSurface) const;

	// declaration only, no definition here:
	template<typename Image>
		requires ValidImageForMode<Mode, Image>
	__host__ std::chrono::duration<float, std::milli> Render(const sf::Vector2u& size, Image& surface, bool moveCamera);

	__host__ ~Renderer();

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

	std::array<Frame, kFramesInFlight> m_Frames;
	uint32_t						   m_SubmittedCount = 0, m_CompletedCount = 0;
};
