#pragma once
#include <memory>

#include "AABB.h"

#include <glm/glm.hpp>

#include "Hittable.h"

class Sphere : public Hittable
{
  public:
	// Templated constructor for material type
	__device__ Sphere(const vec3& center, Float radius, const Material* const material)
		: Hittable(this),
		  m_Center(center),
		  m_Radius(radius),
		  m_BoundingBox(
			  m_Center - m_Radius,
			  m_Center + m_Radius),
		  m_MaterialPtr(material)
	{
	}

	__device__ bool Hit(const Ray& ray, const Float tMin, const Float tMax, HitRecord& record) const
	{
		vec3  oc		   = ray.Origin() - m_Center;
		float a			   = dot(ray.Direction(), ray.Direction());
		float b			   = dot(oc, ray.Direction());
		float c			   = dot(oc, oc) - m_Radius * m_Radius;
		float discriminant = b * b - a * c;
		if (discriminant > 0)
		{
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < tMax && temp > tMin)
			{
				record.T		   = temp;
				record.Location	   = ray.point_at_parameter(record.T);
				record.Normal	   = ((record.Location - m_Center) / m_Radius).make_unit_vector();
				record.MaterialPtr = m_MaterialPtr;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < tMax && temp > tMin)
			{
				record.T		   = temp;
				record.Location	   = ray.point_at_parameter(record.T);
				record.Normal	   = ((record.Location - m_Center) / m_Radius).make_unit_vector();
				record.MaterialPtr = m_MaterialPtr;
				return true;
			}
		}
		return false;
	}

	__device__ bool GetBoundingBox(double time0, double time1, AABB& outputBox) const
	{
		outputBox = m_BoundingBox;
		return true;
	}

	vec3				  m_Center;
	Float				  m_Radius;
	AABB				  m_BoundingBox;
	const Material* const m_MaterialPtr;
};
