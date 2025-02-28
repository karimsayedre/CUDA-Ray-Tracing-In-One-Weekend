#pragma once
#include "Material.h"
#include "Random.h"

//class u_Metal : public Material
//{
//public:
//	__device__ u_Metal(const vec3& color, const Float fuzz)
//		:  Albedo(color), m_Fuzz(fuzz)
//	{
//	}
//
//	__device__ bool Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const override
//	{
//		const vec3 reflected = reflect((ray.Direction().make_unit_vector()), rec.Normal);
//		scattered = { rec.Location, reflected + m_Fuzz * (vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))).make_unit_vector() };
//		attenuation = Albedo;
//		return dot(scattered.Direction(), rec.Normal) > 0.0f;
//	}
//
//	vec3 Albedo;
//	Float m_Fuzz;
//};

