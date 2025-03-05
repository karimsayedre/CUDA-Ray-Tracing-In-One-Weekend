#pragma once
#include "Hittable.h"
#include "Random.h"
#include "Ray.h"
#include "Vec3.h"

enum class MaterialType
{
	Lambert,
	Metal,
	Dielectric,
};

class Material
{
	MaterialType m_Type;

	// union
	//{
	// struct
	//{
	glm::vec3 m_Albedo;
	float	  m_Fuzz;
	//};

	float m_IOR;
	//};

  protected:
	/*union
	{
		u_Lambert*	u_Lambert;
		u_Dielectric* u_Dielectric;
		u_Metal*		u_Metal;
	};*/

  public:
	//__device__ Material(Material* mat)
	//{
	//}

	__device__ Material(MaterialType type, const glm::vec3& albedo, float fuzz = 0.0f, float ior = 1.0f)
		: m_Type(type), m_Albedo(albedo), m_Fuzz(fuzz), m_IOR(ior)
	{
	}

  public:
	__device__ bool Scatter(Ray& incomingRay, Ray& scatteredRay, const HitRecord& rec, glm::vec3& attenuation, uint32_t& randSeed) const;
};

// Optimized Material Scatter Method
__device__ inline bool Material::Scatter(Ray& incomingRay, Ray& scatteredRay, const HitRecord& rec, glm::vec3& attenuation, uint32_t& randSeed) const
{
	// Use faster approximations and reduce branches
	const glm::vec3 rayDir	  = incomingRay.Direction();
	const float		rayDirLen = fastLength(rayDir);
	const glm::vec3 unitDir	  = (rayDirLen > 0.f) ? (rayDir / rayDirLen) : rayDir;

	switch (m_Type)
	{
		case MaterialType::Lambert:
		{
			// Optimize random vector generation
			glm::vec3 scatterDir = rec.Normal + RandomVec3(randSeed);

			// Prevent degenerate scatter directions
			scatterDir = glm::length(scatterDir) > 0.001f
							 ? glm::normalize(scatterDir)
							 : rec.Normal;

			scatteredRay = Ray(rec.Location, scatterDir);
			attenuation *= m_Albedo;
			return true;
		}

		case MaterialType::Metal:
		{
			// Combine reflection and fuzz in one step
			glm::vec3 reflected = reflect(unitDir, rec.Normal);
			glm::vec3 fuzzDir	= RandomVec3(randSeed) * m_Fuzz;

			scatteredRay = Ray(rec.Location, reflected + fuzzDir);
			attenuation *= m_Albedo;

			return dot(scatteredRay.Direction(), rec.Normal) > 0.0f;
		}

		case MaterialType::Dielectric:
		{
			// Reduce repeated calculations
			const float dotRayNormal = dot(unitDir, rec.Normal);
			const bool	frontFace	 = (dotRayNormal < 0.f);

			const glm::vec3 outwardNormal = frontFace ? rec.Normal : -rec.Normal;
			const float		niOverNt	  = frontFace ? (1.f / m_IOR) : m_IOR;
			const float		cosine		  = frontFace ? -dotRayNormal : dotRayNormal;

			glm::vec3 refracted;
			float	  reflectProb;

			// Combine refraction and reflection checks
			if (refract(unitDir, outwardNormal, niOverNt, refracted))
			{
				reflectProb = Reflectance(cosine, m_IOR);
			}
			else
			{
				reflectProb = 1.0f;
				// Force reflection if total internal reflection
				scatteredRay = Ray(rec.Location, reflect(unitDir, outwardNormal));
				return true;
			}

			// Simplified probabilistic scattering
			scatteredRay = (RandomFloat(randSeed) < reflectProb)
							   ? Ray(rec.Location, reflect(unitDir, outwardNormal))
							   : Ray(rec.Location, refracted);

			return true;
		}

		default:
			return false;
	}
}
