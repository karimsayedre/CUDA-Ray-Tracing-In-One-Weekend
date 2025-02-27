#pragma once
#include "Material.h"
#include "Random.h"

class Metal : public Material
{
public:
	__device__ Metal(const vec3& color, const Float fuzz)
		: Material(this), m_Albedo(color), m_Fuzz(fuzz)
	{
	}

	__device__ bool Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
	{
		const vec3 reflected = reflect((ray.Direction().make_unit_vector()), rec.Normal);
		scattered = { rec.Location, reflected + m_Fuzz * (vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))).make_unit_vector() };
		attenuation = m_Albedo;
		return dot(scattered.Direction(), rec.Normal) > 0.0f;
	}

	vec3 m_Albedo;
	Float m_Fuzz;
};

