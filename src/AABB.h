#pragma once

struct __align__(32) AABB
{
	__device__ __host__ AABB()
		: Min(FLT_MAX),
		  Max(-FLT_MAX)
	{
	}

	__device__ __host__ AABB(const Vec3& a, const Vec3& b)
		: Min(glm::min(a, b)),
		  Max(glm::max(a, b))
	{
	}

	__device__ __host__ AABB(const AABB& box0, const AABB& box1)
		: Min(glm::min(box0.Min, box1.Min)),
		  Max(glm::max(box0.Max, box1.Max))
	{
	}

	__device__ __host__ float SurfaceArea() const
	{
		const float dx = Max.x - Min.x;
		const float dy = Max.y - Min.y;
		const float dz = Max.z - Min.z;
		return 2.0f * (dx * dy + dy * dz + dz * dx);
	}

	__device__ __host__ [[nodiscard]] Vec3 Center() const
	{
		return (Min + Max) * 0.5f;
	}

	__align__(16) Vec3 Min;
	__align__(16) Vec3 Max;
};
