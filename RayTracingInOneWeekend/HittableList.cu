#include "pch.cuh"
#include "HittableList.h"

__device__ HittableList::HittableList(Hittable** objects, uint32_t count)
	: Hittable(this), m_Objects(objects), m_Count(count)
{
	// Add(object);
	for (uint32_t i = 0; i < count; i++)
	{
		AABB objectBox = objects[i]->GetBoundingBox(0, 1);
		m_BoundingBox  = AABB(m_BoundingBox, objectBox);
	}
}

__device__ AABB HittableList::GetBoundingBox(double time0, double time1) const
{
	return m_BoundingBox;
}

__device__ bool HittableList::Hit(const Ray& ray, const Float tMin, const Float tMax, HitRecord& record) const
{
	HitRecord tempRecord;
	bool	  hitAnything  = false;
	Float	  closestSoFar = tMax;

	for (uint32_t i = 0; i < m_Count; i++)
	{
		if (m_Objects[i]->Hit(ray, tMin, closestSoFar, tempRecord))
		{
			hitAnything	 = true;
			closestSoFar = tempRecord.T;
			record		 = tempRecord;
		}
	}
	return hitAnything;
}