#pragma once

#include <cuda_runtime.h>
#include "CudaRenderer.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class Camera {
public:
	glm::vec3 m_Origin;
	glm::vec3 m_LowerLeftCorner;
	glm::vec3 m_Horizontal;
	glm::vec3 m_Vertical;

	__host__ __device__ Camera(const glm::vec3& lookFrom, const glm::vec3& lookAt, const glm::vec3& up, const float vFov, const float aspectRatio)
	{
		const auto theta		  = vFov * ((float)M_PI) / 180.0f;
		const auto h = std::tan(theta / 2.0f);
		const auto viewportHeight = 2.0f * h;
		const auto viewportWidth = aspectRatio * viewportHeight;

		const auto w = glm::normalize(lookFrom - lookAt);
		const auto u = glm::normalize(cross(up, w));
		const auto v = glm::cross(w, u);


		m_Origin = lookFrom;
		m_Horizontal = u * viewportWidth;
		m_Vertical = (v * viewportHeight);
		m_LowerLeftCorner = m_Origin - m_Horizontal / 2.0f - m_Vertical / 2.0f - w;
	}

	__device__ Ray GetRay(float u, float v) const {
		return { m_Origin, m_LowerLeftCorner + u * m_Horizontal + v * m_Vertical - m_Origin };
	}
};
