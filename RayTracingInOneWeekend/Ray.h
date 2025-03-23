#pragma once
#include <cuda_runtime.h>
#include "Vec3.h"

class Ray
{
  public:
	Ray() noexcept = default;
	[[nodiscard]] __device__ Ray(const Vec3& origin, const Vec3& direction) noexcept
		: m_Origin(origin), m_Direction(direction)//, m_InvDirection(1.0f / direction)
	{
	}

	[[nodiscard]] __device__ Vec3 Origin() const noexcept
	{
		return m_Origin;
	}
	[[nodiscard]] __device__ Vec3 Direction() const noexcept
	{
		return m_Direction;
	}

	[[nodiscard]] __device__ Vec3 point_at_parameter(float t) const
	{
		return m_Origin + t * m_Direction;
	}

	//[[nodiscard]] __device__ Vec3 InverseDirection() const
	//{
	//	return m_InvDirection;
	//}

private:
	Vec3 m_Origin;
	Vec3 m_Direction;
	//Vec3 m_InvDirection;
};
