#pragma once

#include "Hittable.h"
#include <algorithm> // Add this include for std::min and std::max
#include "AABB.h"
#include <math.h>
class HittableList : public Hittable
{
  public:
	//__device__ HittableList()
	//	: Hittable(this)
	//{
	//}
	__device__ HittableList(Hittable** objects, uint32_t count);

	//__device__ void Clear() { m_Objects.clear(); }
	//__device__ void Add(Hittable* object) { m_Objects.push_back(object); }
	__device__ bool               Hit(const Ray& ray, const Float tMin, const Float tMax, HitRecord& record) const override;
	__device__ [[nodiscard]] AABB GetBoundingBox(double time0, double time1) const override;

	Hittable** m_Objects;
	uint32_t   m_Count;
	AABB	   m_BoundingBox;
};

//__device__ inline AABB SurroundingBox(const AABB& box0, const AABB& box1)
//{
//	vec3 minimum(std::min(box0.().x(), box1.Min().x()),
//				 std::min(box0.Min().y(), box1.Min().y()),
//				 std::min(box0.Min().z(), box1.Min().z()));
//
//	vec3 maximum(std::max(box0.Max().x(), box1.Max().x()),
//				 std::max(box0.Max().y(), box1.Max().y()),
//				 std::max(box0.Max().z(), box1.Max().z()));
//
//	return {minimum, maximum};
//}
