#pragma once
#include <memory>

#include "AABB.h"

#include <glm/glm.hpp>

#include "Hittable.h"
#include "Material.h"

class Sphere : public Hittable
{
  public:
	// Templated constructor for material type
	__device__ Sphere(const glm::vec3& center, Float radius, const uint16_t& materialIndex)
		: m_Center(center),
		  m_Radius(radius),
		  m_BoundingBox(
			  m_Center - m_Radius,
			  m_Center + m_Radius),
		  m_MaterialIndex(materialIndex)
	{
	}

	__device__ __noinline__ bool Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record) const override
	{
		glm::vec3 oc		   = ray.Origin() - m_Center;
		float	  a			   = dot(ray.Direction(), ray.Direction());
		float	  b			   = dot(oc, ray.Direction());
		float	  c			   = dot(oc, oc) - m_Radius * m_Radius;
		float	  discriminant = b * b - a * c;
		if (discriminant > 0)
		{
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < tMax && temp > tMin)
			{
				record.T		   = temp;
				record.Location	   = ray.point_at_parameter(record.T);
				record.Normal	   = glm::normalize((record.Location - m_Center) / m_Radius);
				record.MaterialIndex = m_MaterialIndex;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < tMax && temp > tMin)
			{
				record.T		   = temp;
				record.Location	   = ray.point_at_parameter(record.T);
				record.Normal	   = glm::normalize((record.Location - m_Center) / m_Radius);
				record.MaterialIndex = m_MaterialIndex;
				return true;
			}
		}
		return false;
	}

	__device__ const AABB& GetBoundingBox(double time0, double time1) const override
	{
		return m_BoundingBox;
	}

	__device__ [[nodiscard]] bool IsLeaf() const override
	{
		return true;
	}

	glm::vec3 m_Center;
	Float	  m_Radius;
	AABB	  m_BoundingBox;
	uint16_t  m_MaterialIndex;
	// Material m_Material;
};
