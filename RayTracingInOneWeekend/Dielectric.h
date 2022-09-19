#pragma once
#include "Material.h"
#include "Random.h"

class Dielectric : public Material
{
public:
	Dielectric(const T ior)
		: m_IOR(ior)
	{
	}

	bool Scatter(const Ray& ray, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered) const override
	{
		attenuation = glm::vec3{ 1.0f };
		const T refractionRatio = rec.FrontFace ? (1.0f / m_IOR) : m_IOR;
		const glm::vec3 direction = glm::normalize(ray.Direction());

		const T cosTheta = std::min(glm::dot(-direction, rec.Normal), 1.0f);
		const T sinTheta = math::Sqrt(1.0f - cosTheta * cosTheta);

		glm::vec3 reflectedDir;
		const bool cannotReflect = refractionRatio * sinTheta > 1.0f;
		if (cannotReflect || reflectance(cosTheta, refractionRatio) > RandomFloat())
			reflectedDir = glm::reflect(direction, rec.Normal);
		else
			reflectedDir = glm::refract(direction, rec.Normal, refractionRatio);
			
		scattered = { rec.Location, reflectedDir };
		return true;
	}

public:
	T m_IOR;

private:
	static T reflectance(const T cosine, const T ior)
	{
		// Use Schlick's approximation for reflectance.
		auto r0 = (1.0f - ior) / (1.0f + ior);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
	}
};

