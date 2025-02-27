#pragma once
#include "Material.h"
#include "Random.h"
#include <EASTL/algorithm.h>

#include "CudaRenderer.cuh"

__device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
	vec3  uv           = unit_vector(v);
	float dt           = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0)
	{
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}


class Dielectric : public Material
{
public:
	__device__ Dielectric(const Float ior)
		: Material(this), m_IOR(ior)
	{
	}

	//__device__ Dielectric() noexcept : Material(this), m_IOR(1.0f) {}



	__device__ bool Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
	{
		vec3  outward_normal;
		vec3  reflected = reflect(ray.Direction(), rec.Normal);
		float ni_over_nt;
		attenuation = vec3(1.0, 1.0, 1.0);
		vec3  refracted;
		float reflect_prob;
		float cosine;
		//const Float refractionRatio = rec.FrontFace ? (1.0f / m_IOR) : m_IOR;

		if (dot(ray.Direction(), rec.Normal) > 0.0f)
		{
			outward_normal = -rec.Normal;
			ni_over_nt	   = m_IOR;
			cosine		   = dot(ray.Direction(), rec.Normal) / ray.Direction().length();
			cosine		   = sqrt(1.0f - m_IOR * m_IOR * (1 - cosine * cosine));
		}
		else
		{
			outward_normal = rec.Normal;
			ni_over_nt	   = 1.0f / m_IOR;
			cosine		   = -dot(ray.Direction(), rec.Normal) / ray.Direction().length();
		}
		if (refract(ray.Direction(), outward_normal, ni_over_nt, refracted))
			reflect_prob = Reflectance(cosine, m_IOR);
		else
			reflect_prob = 1.0f;
		if (curand_uniform(local_rand_state) < reflect_prob)
			scattered = Ray(rec.Location, reflected);
		else
			scattered = Ray(rec.Location, refracted);
		return true;
	}

public:
	Float m_IOR;

private:
	__device__ static Float Reflectance(const Float cosine, const Float ior)
	{
		// Use Schlick's approximation for reflectance.
		auto r0 = (1.0f - ior) / (1.0f + ior);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1.0f - cosine), 5.0f);
	}
};

