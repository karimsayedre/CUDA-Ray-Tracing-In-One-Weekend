#pragma once
#include <any>
#include <glm/vec3.hpp>

#include "Vec3.h"
#include "Ray.h"


class BVHNode;
class HittableList;
class Sphere;
class AABB;
class Material;

struct HitRecord
{
	glm::vec3 Location;
	glm::vec3 Normal;
	T T;
	Material* MaterialPtr;
	bool FrontFace;
	inline void SetFaceNormal(const Ray& ray, const glm::vec3& outwardNormal)
	{
		FrontFace = glm::dot(ray.Direction(), outwardNormal) < 0.0f;
		Normal = FrontFace ? outwardNormal : -outwardNormal;
	}
};

enum class HittableType
{
	Invalid,
	BvhNode,
	HittableList,
	Sphere,
};




class Hittable
{
	HittableType Type;
	void* Derived;

	
public:

	Hittable(Sphere* hittable)
		: Type(HittableType::Sphere), Derived(hittable)
	{
	}

	Hittable(HittableList* hittable)
		: Type(HittableType::HittableList), Derived(hittable)
	{
	}

	Hittable(BVHNode* hittable)
		: Type(HittableType::BvhNode), Derived(hittable)
	{
	}


	Hittable() = delete;
	bool Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const;

	bool BoundingBox(double time0, double time1, AABB& outputBox) const;
};


