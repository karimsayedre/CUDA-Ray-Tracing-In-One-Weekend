#pragma once
#include <EASTL/shared_ptr.h>

#include "Hittable.h"
#include "AABB.h"


class Sphere : public Hittable
{
public:
    //constexpr HittableType s_HittableType = HittableType::Sphere;

	Sphere() noexcept : Hittable(this) {}
	Sphere(const glm::vec3& center, const T radius, const eastl::shared_ptr<Material> material) noexcept
	: Hittable(this), m_Center(center), m_Radius(radius), m_MaterialPtr(material)
	{
	}

	bool Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const
    {
        glm::vec3 oc = ray.Origin() - m_Center;
        auto dir = ray.Direction();
        auto a = glm::dot(dir, dir);
        auto half_b = glm::dot(oc, dir);
        auto c = glm::dot(oc, oc) - m_Radius * m_Radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root < tMin || tMax < root) {
            root = (-half_b + sqrtd) / a;
            if (root < tMin || tMax < root)
                return false;
        }

        record.T = root;
        record.Location = ray.At(record.T);
        record.SetFaceNormal(ray, (record.Location - m_Center) / m_Radius);
        record.MaterialPtr = m_MaterialPtr.get();
        return true;

    }
	bool BoundingBox(double time0, double time1, AABB& outputBox) const
    {
        outputBox = AABB(
            m_Center - glm::vec3(m_Radius, m_Radius, m_Radius),
            m_Center + glm::vec3(m_Radius, m_Radius, m_Radius));
        return true;
    }

	glm::vec3 m_Center;
	T m_Radius;
	eastl::shared_ptr<Material> m_MaterialPtr;


};



