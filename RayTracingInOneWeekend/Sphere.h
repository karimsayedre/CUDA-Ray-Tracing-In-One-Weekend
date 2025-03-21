#pragma once
#include "Material.h"

class Sphere 
{
  public:
	// Templated constructor for material type
	__device__ Sphere(const Vec3& center, Float radius, const uint16_t& materialIndex);

	__device__ bool Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record) const;
	//__device__ AABB			GetBoundingBox() const
	//{
	//	return m_BoundingBox;
	//}
	Vec3 m_Center;
	Float	  m_Radius;
	uint16_t  m_MaterialIndex;
	//AABB	  m_BoundingBox;
	// Material m_Material;
};
