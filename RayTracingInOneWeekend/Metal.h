#pragma once
#include "Material.h"

class Metal : public Material
{
public:
	Metal(const glm::vec3& color, const T fuzz)
		: m_Albedo(color), m_Fuzz(fuzz)
	{
	}

	bool Scatter(const Ray& ray, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered) const override
	{
		auto seed = (uint32_t)rand();
		const glm::vec3 reflected = glm::reflect(glm::normalize(ray.Direction()), rec.Normal);
		scattered = { rec.Location, reflected + m_Fuzz * RandomNormalizedVector(seed) };
		attenuation = m_Albedo;
		return glm::dot(scattered.Direction(), rec.Normal) > 0.0f;
	}

	glm::vec3 m_Albedo;
	T m_Fuzz;
};

