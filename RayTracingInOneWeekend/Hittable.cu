#include "pch.cuh"

#include "Hittable.h"
#include "BVH.h"

__device__ bool Hittable::Hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const
{
	switch (Type)
	{
		case HittableType::eBvhNode: return u_BvhNode->Hit(ray, tMin, tMax, record);
		case HittableType::eHittableList: return u_HittableList->Hit(ray, tMin, tMax, record);
		case HittableType::eSphere: return u_Sphere->Hit(ray, tMin, tMax, record);
	}
	return false;
}