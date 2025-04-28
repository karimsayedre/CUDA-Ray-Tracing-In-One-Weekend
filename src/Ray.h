#pragma once

class Ray
{
  public:
	Ray() = default;

	[[nodiscard]] __device__ __host__ Ray(const Vec3& origin, const Vec3& direction) noexcept
		: Origin(origin), Direction(direction)
	{
	}

		[[nodiscard]] __device__ __host__ Ray(const float3& origin, const float3& direction) noexcept
		: Origin(origin.x, origin.y, origin.z), Direction(direction.x, direction.y, direction.z)
	{
	}

	[[nodiscard]] __device__ __host__ Vec3 PointAtT(const float t) const noexcept
	{
		return Origin + t * Direction;
	}

	Vec3 Origin;
	Vec3 Direction;
};
