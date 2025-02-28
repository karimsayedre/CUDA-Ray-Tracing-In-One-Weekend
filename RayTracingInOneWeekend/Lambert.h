#pragma once
#include <iostream>
#include <ostream>

#include "Material.h"
#include "Random.h"

//class u_Lambert : public Material
//{
//  public:
//	__device__ u_Lambert(const vec3& color)
//		: Albedo(color)
//	{
//	}
//
//	__device__ bool Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const override
//	{
//		vec3 scatterDirection = (vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))).make_unit_vector();
//		// printf("x= %.3f, y= %.3f, z= %.3f \n", scatterDirection.x, scatterDirection.y, scatterDirection.z);
//		scattered	= Ray(rec.Location, rec.Normal + scatterDirection);
//		attenuation = Albedo;
//		return true;
//	}
//
//	vec3 Albedo;
//};
