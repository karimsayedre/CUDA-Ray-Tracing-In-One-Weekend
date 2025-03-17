#pragma once
#include <cuda_runtime.h>
#include "Vec3.h"

class Ray
{
  public:
	Ray() noexcept = default;
	[[nodiscard]] __device__ Ray(const glm::vec3& origin, const glm::vec3& direction) noexcept
		: m_Origin(origin), m_Direction(direction), m_InvDirection(1.0f / direction)
	{
	}

	[[nodiscard]] __device__ glm::vec3 Origin() const noexcept
	{
		return m_Origin;
	}
	[[nodiscard]] __device__ glm::vec3 Direction() const noexcept
	{
		return m_Direction;
	}

	[[nodiscard]] __device__ glm::vec3 point_at_parameter(float t) const
	{
		return m_Origin + t * m_Direction;
	}

	[[nodiscard]] __device__ glm::vec3 InverseDirection() const
	{
		return m_InvDirection;
	}

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
	glm::vec3 m_InvDirection;
};
