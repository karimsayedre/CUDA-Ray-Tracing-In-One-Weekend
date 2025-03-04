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

//__device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
//{
//	vec3  uv		   = unit_vector(v);
//	float dt		   = dot(uv, n);
//	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
//	if (discriminant > 0)
//	{
//		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
//		return true;
//	}
//	else
//		return false;
//}

//__device__ inline vec3 reflect(const vec3& v, const vec3& n)
//{
//	return v - 2.0f * dot(v, n) * n;
//}

class Material
{
	MaterialType m_Type;

	// union
	//{
	// struct
	//{
	vec3  m_Albedo;
	float m_Fuzz;
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

	__device__ Material(MaterialType type, const vec3& albedo, float fuzz = 0.0f, float ior = 1.0f)
		: m_Type(type), m_Albedo(albedo), m_Fuzz(fuzz), m_IOR(ior)
	{
	}

  public:
	__device__ bool Scatter(Ray& ray, const HitRecord& rec, vec3& attenuation, uint32_t& randSeed) const;
};

//__device__ static Float Reflectance(const Float cosine, const Float ior)
//{
//	// Use Schlick's approximation for reflectance.
//	auto r0 = (1.0f - ior) / (1.0f + ior);
//	r0		= r0 * r0;
//	return r0 + (1 - r0) * pow((1.0f - cosine), 5.0f);
//}

__device__ inline bool Material::Scatter(
	Ray&			 ray,
	const HitRecord& rec,
	vec3&			 attenuation,
	uint32_t&		 randSeed) const
{
	// Cache these to avoid repeated calls
	vec3  rayDir	= ray.Direction(); // local copy
	float rayDirLen = fastLength(rayDir);
	// If using exact length(), do: float rayDirLen = length(rayDir);

	// Normalize once if needed
	vec3 unitDir = (rayDirLen > 0.f) ? (rayDir / rayDirLen) : rayDir;

	switch (m_Type)
	{
		// -----------------------------------------------------------
		// Lambert
		// -----------------------------------------------------------
		case MaterialType::Lambert:
		{
			// Generate random unit direction
			vec3 scatterDir = rec.Normal + randomUnitVector(randSeed);

			// Update the ray
			ray = Ray(rec.Location, scatterDir);

			// Modulate attenuation
			attenuation *= m_Albedo;
			return true;
		}

		// -----------------------------------------------------------
		// Metal
		// -----------------------------------------------------------
		case MaterialType::Metal:
		{
			// Reflect the *normalized* direction
			vec3 reflected = reflect(unitDir, rec.Normal);

			// Add fuzz
			vec3 fuzzDir = randomUnitVector(randSeed) * m_Fuzz;

			ray = Ray(rec.Location, reflected + fuzzDir);

			// Modulate attenuation
			attenuation *= m_Albedo;

			// Only scatter if the ray is still above the surface
			return (dot(ray.Direction(), rec.Normal) > 0.0f);
		}

		// -----------------------------------------------------------
		// Dielectric (Refraction)
		// -----------------------------------------------------------
		case MaterialType::Dielectric:
		{
			// Figure out if we're inside or outside
			float dotRayNormal = dot(unitDir, rec.Normal);
			bool  frontFace	   = (dotRayNormal < 0.f);

			// Adjust normal & refraction ratio
			vec3  outwardNormal = frontFace ? rec.Normal : -rec.Normal;
			float niOverNt		= frontFace ? (1.f / m_IOR) : m_IOR;
			float cosine		= frontFace ? -dotRayNormal : dotRayNormal;

			// Attempt to refract
			vec3  refracted;
			float reflectProb;
			if (refract(unitDir, outwardNormal, niOverNt, refracted))
				reflectProb = Reflectance(cosine, m_IOR);
			else
				reflectProb = 1.0f;

			// Randomly reflect or refract
			if (RandomFloat(randSeed) < reflectProb)
			{
				vec3 reflected = reflect(unitDir, outwardNormal);
				ray			   = Ray(rec.Location, reflected);
			}
			else
			{
				ray = Ray(rec.Location, refracted);
			}
			return true;
		}
	}
	return false;
}
