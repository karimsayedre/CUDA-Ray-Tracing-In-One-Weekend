#pragma once

#include "Hittable.h"
#include "Sphere.h"

struct HitRecord;

class HittableList
{
  public:
	HittableList() = delete;

	__device__ void SetAABBs() const
	{
		for (uint32_t i = 0; i < m_Count; i++)
		{
			m_AABB[i] = AABB(m_Objects[i].m_Center - m_Objects[i].m_Radius, m_Objects[i].m_Center + m_Objects[i].m_Radius);
		}
	}

	__device__ bool Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record) const
	{
		HitRecord tempRecord;
		bool	  hitAnything = false;
		// Float	  closestSoFar = tMax;

		for (uint32_t i = 0; i < m_Count; i++)
		{
			if (m_Objects[i].Hit(ray, tMin, tMax, tempRecord))
			{
				hitAnything = true;
				tMax		= tempRecord.T;
				record		= tempRecord;
			}
		}
		return hitAnything;
	}

	__device__ bool HitSphere(uint32_t sphereIndex, const Ray& ray, Float tmin, Float& closestSoFar, HitRecord& hitRecord) const
	{
		return m_Objects[sphereIndex].Hit(ray, tmin, closestSoFar, hitRecord);
	}

	// private:

	Sphere*	 m_Objects;
	AABB*	 m_AABB;
	uint32_t m_Count;
};

//__device__ inline AABB SurroundingBox(const AABB& box0, const AABB& box1)
//{
//	vec3 minimum(std::Min(box0.().x(), box1.Min().x()),
//				 std::Min(box0.Min().y(), box1.Min().y()),
//				 std::Min(box0.Min().z(), box1.Min().z()));
//
//	vec3 maximum(std::Max(box0.Max().x(), box1.Max().x()),
//				 std::Max(box0.Max().y(), box1.Max().y()),
//				 std::Max(box0.Max().z(), box1.Max().z()));
//
//	return {minimum, maximum};
//}
