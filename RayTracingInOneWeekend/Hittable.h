#pragma once
#include "Ray.h"

class Material;

struct HitRecord
{
	glm::vec3 Location;
	glm::vec3 Normal;
	T T;
	std::shared_ptr<Material> MaterialPtr;
	bool FrontFace;
	inline void SetFaceNormal(const Ray& ray, const glm::vec3& outwardNormal)
	{
		FrontFace = glm::dot(ray.Direction(), outwardNormal) < 0.0f;
		Normal = FrontFace ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	virtual bool Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const = 0;
};

