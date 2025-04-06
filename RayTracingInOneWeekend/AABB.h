#pragma once

struct alignas(32) AABB
{
	__device__ AABB()
		: Min(std::numeric_limits<float>::max()),
		  Max(-std::numeric_limits<float>::max())
	{
	}

	__device__ AABB(const Vec3& a, const Vec3& b)
		: Min(glm::min(a, b)),
		  Max(glm::max(a, b))
	{
	}

	__device__ AABB(const AABB& box0, const AABB& box1)
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

	__device__ [[nodiscard]] Vec3 Center() const
	{
		return (Min + Max) * 0.5f;
	}

	Vec3 Min;
	Vec3 Max;
};

__device__ inline AABB operator+(const AABB& box, const Vec3& offset)
{
	return {box.Min + offset, box.Max + offset};
}

__device__ inline AABB operator+(const Vec3& offset, const AABB& box)
{
	return box + offset;
}
