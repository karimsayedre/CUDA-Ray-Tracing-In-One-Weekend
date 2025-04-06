#pragma once

class Ray
{
  public:
	[[nodiscard]] __device__ Ray(const Vec3& origin, const Vec3& direction) noexcept
		: m_Origin(origin), m_Direction(direction)
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

	[[nodiscard]] __device__ Vec3 PointAtT(const float t) const noexcept
	{
		return m_Origin + t * m_Direction;
	}

private:
	Vec3 m_Origin;
	Vec3 m_Direction;
};
