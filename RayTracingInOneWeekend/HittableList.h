#pragma once

#include "Hittable.h"
#include "Sphere.h"

struct HitRecord;

class HittableList : public Hittable
{
  public:
	__device__ HittableList(Hittable** objects, uint32_t count)
		: Hittable(this), m_Objects(objects), m_Count(count)
	{
		// Add(object);
		for (uint32_t i = 0; i < count; i++)
		{
			AABB objectBox = objects[i]->GetBoundingBox(0, 1);
			m_BoundingBox  = AABB(m_BoundingBox, objectBox);
		}
	}


	__device__ bool Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record) const override
	{
		HitRecord tempRecord;
		bool	  hitAnything  = false;
		//Float	  closestSoFar = tMax;

		for (uint32_t i = 0; i < m_Count; i++)
		{
			if ((Sphere*)m_Objects[i]->Hit(ray, tMin, tMax, tempRecord))
			{
				hitAnything	 = true;
				tMax		= tempRecord.T;
				record		 = tempRecord;
			}
		}
		return hitAnything;
	}

  //private:

	Hittable** m_Objects;
	uint32_t   m_Count;
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
