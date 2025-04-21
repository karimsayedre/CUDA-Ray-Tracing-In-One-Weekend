#pragma once
#include "Ray.h"

///// WARNING -- DATA MUST BE THE SAME AS IN THE CAMERA STRUCT BELOW
class Camera
{
  public:
	__device__ __host__ Camera(const Vec3& lookFrom, const Vec3& lookAt, const Vec3& up, const float vFov, float width, float height)
		: m_LookAt(lookAt), m_OriginalOffset(lookFrom.x)
	{
		const float theta		   = vFov * std::numbers::pi_v<float> / 180.0f;
		const float h			   = glm::tan(theta / 2.0f);
		const float viewportHeight = 2.0f * h;

		float		aspectRatio	  = width / height;
		const float viewportWidth = aspectRatio * viewportHeight;

		const Vec3 w = glm::normalize(lookFrom - m_LookAt);
		const Vec3 u = glm::normalize(cross(up, w));
		const Vec3 v = glm::cross(w, u);

		m_Origin		  = lookFrom;
		m_Horizontal	  = u * viewportWidth;
		m_Vertical		  = v * viewportHeight;
		m_LowerLeftCorner = m_Origin - m_Horizontal / 2.0f - m_Vertical / 2.0f - w;
	}

	__device__ __host__ void ResizeViewport(const float newAspectRatio)
	{
		const float currentViewportHeight = glm::length(m_Vertical);

		const float newViewportWidth = newAspectRatio * currentViewportHeight;

		const Vec3 w = glm::normalize(m_Origin - m_LookAt);
		const Vec3 u = glm::normalize(m_Horizontal);

		m_Horizontal = u * newViewportWidth;

		m_LowerLeftCorner = m_Origin - m_Horizontal / 2.0f - m_Vertical / 2.0f - w;
	}

	__device__ __host__ void MoveAndLookAtSamePoint(const Vec3& offset, const float resetPoint)
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

	__device__ __host__ [[nodiscard]] Ray GetRay(const float2 uv) const
	{
		return {m_Origin, m_LowerLeftCorner + uv.x * m_Horizontal + uv.y * m_Vertical - m_Origin};
	}

	///// WARNING -- DATA MUST BE THE SAME AS IN THE CAMERA STRUCT BELOW
  public:
	Vec3  m_Origin;
	Vec3  m_LowerLeftCorner;
	Vec3  m_Horizontal;
	Vec3  m_Vertical;
	Vec3  m_LookAt;
	float m_OriginalOffset;
};

///// WARNING -- DATA MUST BE THE SAME AS IN THE CAMERA CLASS
struct CameraPOD
{
	float3 m_Origin;
	float3 m_LowerLeftCorner;
	float3 m_Horizontal;
	float3 m_Vertical;
	float3 m_LookAt;
	float  m_OriginalOffset;
};
