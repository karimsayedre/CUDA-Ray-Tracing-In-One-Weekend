#include "Hittable.h"
#include "Sphere.h"
#include "BVH.h"


bool Hittable::Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const
{
		
	switch (Type)
	{
	case HittableType::BvhNode: return static_cast<const BVHNode*>(Derived)->Hit(ray, tMin, tMax, record);
	case HittableType::HittableList:return static_cast<const HittableList*>(Derived)->Hit(ray, tMin, tMax, record);
	case HittableType::Sphere: return static_cast<const Sphere*>(Derived)->Hit(ray, tMin, tMax, record);
	}
	assert(false);
	return false;
}

bool Hittable::BoundingBox(double time0, double time1, AABB& outputBox) const
{
	switch (Type)
	{
	case HittableType::BvhNode: return static_cast<const BVHNode*>(Derived)->BoundingBox(time0, time1, outputBox);
	case HittableType::HittableList:return static_cast<const HittableList*>(Derived)->BoundingBox(time0, time1, outputBox);
	case HittableType::Sphere: return static_cast<const Sphere*>(Derived)->BoundingBox(time0, time1, outputBox);
	}
	assert(false);
	return false;
}
