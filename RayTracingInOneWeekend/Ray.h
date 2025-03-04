#pragma once
#include <cuda_runtime.h>
#include "Vec3.h"

class Ray
{
  public:
	//[[nodiscard]] __device__ Ray() noexcept
	//{
	//}
	[[nodiscard]] __device__ Ray(const vec3& origin, const vec3& direction) noexcept
		: m_Origin(origin), m_Direction(direction)/*, m_InvDirection(1.0f / direction)*/
	{
	}

	[[nodiscard]] __device__ vec3 Origin() const noexcept
	{
		return m_Origin;
	}
	[[nodiscard]] __device__ vec3 Direction() const noexcept
	{
		return m_Direction;
	}
	//[[nodiscard]] __device__ vec3 InvDirection() const noexcept
	//{
	//	return m_InvDirection;
	//}

	[[nodiscard]] __device__ vec3 point_at_parameter(float t) const
	{
		return m_Origin + t * m_Direction;
	}

	[[nodiscard]] __device__ vec3 At(const Float t) const noexcept
	{
		return m_Origin + t * m_Direction;
	}

  private:
	vec3 m_Origin;
	vec3 m_Direction;
	//vec3 m_InvDirection;
};
