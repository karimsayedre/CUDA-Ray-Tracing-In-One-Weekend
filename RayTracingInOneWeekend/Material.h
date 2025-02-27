#pragma once

#include "Hittable.h"
#include "Ray.h"

class Lambert;
class Metal;
class Dielectric;

enum class MaterialType
{
	Lambert,
	Metal,
	Dielectric,
};


__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
	return v - 2.0f * dot(v, n) * n;
}

class Material
{

	MaterialType m_Type;

	union
	{
		Lambert*	u_Lambert;
		Dielectric* u_Dielectric;
		Metal*		u_Metal;
	};

  public:
	Material() = delete;

	__device__ Material(Dielectric* material)
		: m_Type(MaterialType::Dielectric), u_Dielectric(material)
	{
	}

	__device__ Material(Lambert* material)
		: m_Type(MaterialType::Lambert), u_Lambert(material)
	{
	}

	__device__ Material(Metal* material)
		: m_Type(MaterialType::Metal), u_Metal(material)
	{
	}

  public:
	__device__ bool Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const;
};
