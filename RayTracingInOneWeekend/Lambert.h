#pragma once
#include "Material.h"

class Lambert : public Material
{
public:
	Lambert(const glm::vec3& color)
		: m_Albedo(color)
	{
	}

	bool Scatter(const Ray& ray, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered) const override
	{
		glm::vec3 scatterDirection = rec.Normal + RandomNormalizedVector();

		if (math::NearZero(scatterDirection))
			scatterDirection = rec.Normal;

		scattered = { rec.Location, scatterDirection };
		attenuation = m_Albedo;
		return true;
	}

	glm::vec3 m_Albedo;
};

