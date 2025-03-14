#include "pch.cuh"

#include "Sphere.h"

__device__ Sphere::Sphere(const glm::vec3& center, Float radius, const uint16_t& materialIndex)
	: Hittable(this,
			   AABB {center - radius,
					 center + radius}),
	  m_Center(center),
	  m_Radius(radius),
	  m_MaterialIndex(materialIndex)
{
}

__device__ bool Sphere::Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record) const
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
			record.T			 = temp;
			record.Location		 = ray.point_at_parameter(record.T);
			record.Normal		 = glm::normalize((record.Location - m_Center) / m_Radius);
			record.MaterialIndex = m_MaterialIndex;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < tMax && temp > tMin)
		{
			record.T			 = temp;
			record.Location		 = ray.point_at_parameter(record.T);
			record.Normal		 = glm::normalize((record.Location - m_Center) / m_Radius);
			record.MaterialIndex = m_MaterialIndex;
			return true;
		}
	}
	return false;
}