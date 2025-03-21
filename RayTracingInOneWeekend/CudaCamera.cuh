#pragma once

#include <cuda_runtime.h>
#include "CudaRenderer.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class Camera
{
  public:
	Vec3 m_Origin;
	Vec3 m_LowerLeftCorner;
	Vec3 m_Horizontal;
	Vec3 m_Vertical;

	__device__ __host__ Camera(const Vec3& lookFrom, const Vec3& lookAt, const Vec3& up, const Float vFov, const Float aspectRatio)
	{
		const auto theta		  = vFov * ((Float)M_PI) / __float2half(180.0f);
		const Float h			  = glm::tan(theta / __float2half(2.0f));
		const auto	viewportHeight = __float2half(2.0f) * h;
		const auto viewportWidth  = aspectRatio * viewportHeight;

		const auto w = glm::normalize(lookFrom - lookAt);
		const auto u = glm::normalize(cross(up, w));
		const auto v = glm::cross(w, u);

		m_Origin		  = lookFrom;
		m_Horizontal	  = u * viewportWidth;
		m_Vertical		  = v * viewportHeight;
		m_LowerLeftCorner = m_Origin - m_Horizontal / __float2half(2.0f) - m_Vertical / __float2half(2.0f) - w;
	}

	__device__ Ray GetRay(Float u, Float v) const
	{
		return {m_Origin, m_LowerLeftCorner + u * m_Horizontal + v * m_Vertical - m_Origin};
	}
};
