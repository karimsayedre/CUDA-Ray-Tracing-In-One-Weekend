#pragma once
#include "Ray.h"

class Camera
{
  public:
	Vec3				m_Origin;
	Vec3				m_LowerLeftCorner;
	Vec3				m_Horizontal;
	Vec3				m_Vertical;
	Vec3				m_LookAt;
	float				m_OriginalOffset;
	__device__ __host__ Camera(const Vec3& lookFrom, const Vec3& lookAt, const Vec3& up, const float vFov, const float aspectRatio)
		: m_LookAt(lookAt), m_OriginalOffset(lookFrom.x)
	{
		const float theta		   = vFov * std::numbers::pi_v<float> / 180.0f;
		const float h			   = glm::tan(theta / 2.0f);
		const float viewportHeight = 2.0f * h;
		const float viewportWidth  = aspectRatio * viewportHeight;

		const Vec3 w = glm::normalize(lookFrom - m_LookAt);
		const Vec3 u = glm::normalize(cross(up, w));
		const Vec3 v = glm::cross(w, u);

		m_Origin		  = lookFrom;
		m_Horizontal	  = u * viewportWidth;
		m_Vertical		  = v * viewportHeight;
		m_LowerLeftCorner = m_Origin - m_Horizontal / 2.0f - m_Vertical / 2.0f - w;
	}

	__device__ __host__ void Translate(const Vec3& offset, const float resetPoint)
	{
		if (m_Origin.x > resetPoint + m_OriginalOffset)
		{
			m_Origin.x = m_OriginalOffset;
		}
		else
		{
			m_Origin += offset;
		}
		const Vec3 w	  = glm::normalize(m_Origin - m_LookAt);
		m_LowerLeftCorner = m_Origin - m_Horizontal / 2.0f - m_Vertical / 2.0f - w;
	}

	__device__ __host__ Ray GetRay(float u, float v) const
	{
		return {m_Origin, m_LowerLeftCorner + u * m_Horizontal + v * m_Vertical - m_Origin};
	}
};
