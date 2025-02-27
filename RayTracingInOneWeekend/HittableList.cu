#include "pch.cuh"
#include "HittableList.h"

__device__ HittableList::HittableList(Hittable** objects, uint32_t count)
	: Hittable(this), m_Objects(objects), m_Count(count)
{
	// Add(object);
}

__device__ bool HittableList::GetBoundingBox(double time0, double time1, AABB& outputBox) const
{
	// if (m_Objects.empty()) return false;

	AABB tempBox;
	bool firstBox = true;

	for (uint32_t i = 0; i < m_Count; i++)
	{
		if (!m_Objects[i]->GetBoundingBox(time0, time1, tempBox))
			return false;
		outputBox = firstBox ? tempBox : SurroundingBox(outputBox, tempBox);
		firstBox  = false;
	}

	m_BoundingBox = outputBox;

	return true;
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