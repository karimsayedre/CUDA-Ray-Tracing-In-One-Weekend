#pragma once
#include "AABB.h"
#include "Vec3.h"

class Sphere;
class HittableList;
class BVHNode;

struct HitRecord
{
	Vec3 Location;
	Float	  T;
	Vec3 Normal;
	uint16_t  MaterialIndex;
	// bool			FrontFace;
	//__device__ void SetFaceNormal(const Ray& ray, const vec3& outwardNormal)
	//{
	// FrontFace = dot(ray.Direction(), outwardNormal) < 0.0f;
	// Normal	  = FrontFace ? outwardNormal : -outwardNormal;
	//}
};

//enum class HittableType
//{
//	eInvalid,
//	eBvhNode,
//	eHittableList,
//	eSphere,
//};

//class Hittable
//{
//	HittableType Type;
//	union
//	{
//		BVHNode*	  u_BvhNode;
//		HittableList* u_HittableList;
//		Sphere*		  u_Sphere;
//	};
//	
//
//  public:
//	//AABB m_BoundingBox;
//
//	__device__ Hittable(Sphere* hittable, const AABB& aabb)
//		: Type(HittableType::eSphere), u_Sphere(hittable)/*, m_BoundingBox(aabb)*/
//	{
//	}
//
//	__device__ Hittable(HittableList* hittable/*, const AABB& aabb*/)
//		: Type(HittableType::eHittableList), u_HittableList(hittable)//, m_BoundingBox(aabb) set in derived constructor
//	{
//	}
//
//	__device__ Hittable(BVHNode* hittable/*, const AABB& aabb*/)
//		: Type(HittableType::eBvhNode), u_BvhNode(hittable)//, m_BoundingBox(aabb)
//	{
//	}
//
//	// Delete default constructor
//	// Hittable() = delete;
//
//	// Methods
//	__device__ [[nodiscard]] virtual bool Hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const = 0;
//	//__device__ [[nodiscard]] const AABB&  GetBoundingBox(double time0, double time1) const
//	//{
//	//	return m_BoundingBox;
//	//}
//	//__device__ [[nodiscard]] bool IsLeaf() const
//	//{
//	//	return Type == HittableType::eSphere;
//	//}
//
//	virtual ~Hittable() = default;
//
//};
