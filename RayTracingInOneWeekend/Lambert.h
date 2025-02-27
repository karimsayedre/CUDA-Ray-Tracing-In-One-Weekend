#pragma once
#include <iostream>
#include <ostream>

#include "Material.h"
#include "Random.h"

class Lambert : public Material
{
  public:
	__device__ Lambert(const vec3& color)
		: Material(this), m_Albedo(color)
	{
	}

	__device__ bool Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
	{
		vec3 scatterDirection = (vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))).make_unit_vector();
		//printf("x= %.3f, y= %.3f, z= %.3f \n", scatterDirection.x, scatterDirection.y, scatterDirection.z);
		scattered	= Ray(rec.Location, rec.Normal + scatterDirection);
		attenuation = m_Albedo;
		return true;
	}

	vec3 m_Albedo;
};
