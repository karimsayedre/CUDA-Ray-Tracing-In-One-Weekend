#include "pch.cuh"
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h" // Include necessary headers
#include "BVH.h"	// ...and others

__device__ void HitRecord::SetFaceNormal(const Ray& ray, const vec3& outwardNormal)
{
	FrontFace = dot(ray.Direction(), outwardNormal) < 0.0f;
	Normal    = FrontFace ? outwardNormal : -outwardNormal;
}

__device__ Hittable::Hittable(Sphere* hittable): Type(HittableType::eSphere), u_Sphere(hittable)
{
}

__device__ Hittable::Hittable(HittableList* hittable)
	: Type(HittableType::eHittableList), u_HittableList(hittable)
{
}

__device__ Hittable::Hittable(BVHNode* hittable)
	: Type(HittableType::eBvhNode), u_BvhNode(hittable)
{
}

//__device__ bool Hittable::Hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const
//{
//	switch (Type)
//	{
//		case HittableType::eBvhNode: return u_BvhNode->Hit(ray, tMin, tMax, record);
//		case HittableType::eHittableList: return u_HittableList->Hit(ray, tMin, tMax, record);
//		case HittableType::eSphere: return u_Sphere->Hit(ray, tMin, tMax, record);
//	}
//	return false; // No assert in device code; can cause synchronization issues
//}
//
//__device__ bool Hittable::GetBoundingBox(double time0, double time1, AABB& outputBox) const
//{
//	switch (Type)
//	{
//		case HittableType::eBvhNode: return u_BvhNode->GetBoundingBox(time0, time1, outputBox);
//		case HittableType::eHittableList: return u_HittableList->GetBoundingBox(time0, time1, outputBox);
//		case HittableType::eSphere: return u_Sphere->GetBoundingBox(time0, time1, outputBox);
//	}
//	return false;
//}