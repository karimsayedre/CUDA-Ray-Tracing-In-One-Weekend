#pragma once
#include <cuda_runtime.h>
#include "Interval.h"
#include "Ray.h"

class alignas(32) AABB
{
  public:
	Vec3 Min;
	Vec3 Max;

	__device__ AABB()
		: Min(std::numeric_limits<Float>::max()),
		  Max(-std::numeric_limits<Float>::max())
	{
	}

	__device__ AABB(const Interval& ix, const Interval& iy, const Interval& iz)
		: Min(Vec3(ix.min, iy.min, iz.min)),
		  Max(Vec3(ix.max, iy.max, iz.max))
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

	__device__ [[nodiscard]] Vec3 Center() const
	{
		return (Min + Max) * 0.5f;
	}

	__device__ [[nodiscard]] int LongestAxis() const
	{
		// Returns the index of the longest axis of the bounding box.
		if (Max.x - Min.x > Max.y - Min.y)
			return Max.x - Min.x > Max.z - Min.z ? 0 : 2;
		else
			return Max.y - Min.y > Max.z - Min.z ? 1 : 2;
	}
};

__device__ inline AABB operator+(const AABB& bbox, const Vec3& offset)
{
	return {bbox.Min + offset, bbox.Max + offset};
}

__device__ inline AABB operator+(const Vec3& offset, const AABB& bbox)
{
	return bbox + offset;
}
