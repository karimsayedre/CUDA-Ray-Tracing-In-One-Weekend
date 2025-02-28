#pragma once
#include "Material.h"
#include "Random.h"
#include <EASTL/algorithm.h>

#include "CudaRenderer.cuh"



// class u_Dielectric : public Material
//{
//  public:
//	__device__ u_Dielectric(const Float ior)
//		: IOR(ior)
//	{
//	}
//
//	__device__ bool Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const override
//	{
//		vec3  outward_normal;
//		vec3  reflected = reflect(ray.Direction(), rec.Normal);
//		float ni_over_nt;
//		attenuation = vec3(1.0, 1.0, 1.0);
//		vec3  refracted;
//		float reflect_prob;
//		float cosine;
//		// const Float refractionRatio = rec.FrontFace ? (1.0f / IOR) : IOR;
//
//		if (dot(ray.Direction(), rec.Normal) > 0.0f)
//		{
//			outward_normal = -rec.Normal;
//			ni_over_nt	   = IOR;
//			cosine		   = dot(ray.Direction(), rec.Normal) / ray.Direction().length();
//			cosine		   = sqrt(1.0f - IOR * IOR * (1 - cosine * cosine));
//		}
//		else
//		{
//			outward_normal = rec.Normal;
//			ni_over_nt	   = 1.0f / IOR;
//			cosine		   = -dot(ray.Direction(), rec.Normal) / ray.Direction().length();
//		}
//		if (refract(ray.Direction(), outward_normal, ni_over_nt, refracted))
//			reflect_prob = Reflectance(cosine, IOR);
//		else
//			reflect_prob = 1.0f;
//
//		if (curand_uniform(local_rand_state) < reflect_prob)
//			scattered = Ray(rec.Location, reflected);
//		else
//			scattered = Ray(rec.Location, refracted);
//		return true;
//	}
//
//  public:
//	Float IOR;
//
//  private:
//	__device__ static Float Reflectance(const Float cosine, const Float ior)
//	{
//		// Use Schlick's approximation for reflectance.
//		auto r0 = (1.0f - ior) / (1.0f + ior);
//		r0		= r0 * r0;
//		return r0 + (1 - r0) * pow((1.0f - cosine), 5.0f);
//	}
//};
