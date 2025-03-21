#include "pch.cuh"

#include "Sphere.h"

__device__ Sphere::Sphere(const Vec3& center, Float radius, const uint16_t& materialIndex)
	: m_Center(center),
	  m_Radius(radius),
	  m_MaterialIndex(materialIndex)
//,
// m_BoundingBox(center - radius,
//			  center + radius)
{
}

__device__ bool Sphere::Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record) const
{
	Vec3  oc		   = ray.Origin() - m_Center;
	Float a			   = dot(ray.Direction(), ray.Direction());
	Float b			   = dot(oc, ray.Direction());
	Float c			   = glm::dot(oc, oc) - m_Radius * m_Radius;
	Float discriminant = b * b - a * c;
	if (discriminant > __float2half(0.0f))
	{
		Float temp = (-b - glm::sqrt(discriminant)) / a;
		if (temp < tMax && temp > tMin)
		{
			record.T			 = temp;
			record.Location		 = ray.point_at_parameter(record.T);
			record.Normal		 = glm::normalize((record.Location - m_Center) / m_Radius);
			record.MaterialIndex = m_MaterialIndex;
			return true;
		}
		temp = (-b + glm::sqrt(discriminant)) / a;
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