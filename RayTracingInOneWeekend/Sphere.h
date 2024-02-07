#pragma once
#include <EASTL/shared_ptr.h>

#include "Hittable.h"

class Sphere : public Hittable
{
public:
	Sphere() noexcept {}
	Sphere(const glm::vec3& center, const T radius, const eastl::shared_ptr<Material> material) noexcept
	: m_Center(center), m_Radius(radius), m_MaterialPtr(material)
	{
	}

	virtual bool Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const override;
	bool BoundingBox(double time0, double time1, AABB& outputBox) const override;

	glm::vec3 m_Center;
	T m_Radius;
	eastl::shared_ptr<Material> m_MaterialPtr;
};



