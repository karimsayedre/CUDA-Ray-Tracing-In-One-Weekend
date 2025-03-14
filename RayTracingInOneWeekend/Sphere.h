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
	__device__ Sphere(const glm::vec3& center, Float radius, const uint16_t& materialIndex);

	__device__ bool Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record) const override;

	glm::vec3 m_Center;
	Float	  m_Radius;
	uint16_t  m_MaterialIndex;
	// Material m_Material;
};
