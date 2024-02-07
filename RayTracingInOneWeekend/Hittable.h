#pragma once
#include "AABB.h"
#include "Ray.h"


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
public:
	Hittable() = default;
	virtual bool Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const = 0;
	virtual bool BoundingBox(double time0, double time1, AABB& outputBox) const = 0;
};


