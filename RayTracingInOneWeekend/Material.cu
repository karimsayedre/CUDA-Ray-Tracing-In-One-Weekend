#include "pch.cuh"
#include "Material.h"


#include "Dielectric.h"
#include "Lambert.h"
#include "Metal.h"




__device__ bool Material::Scatter(const Ray& ray, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
{
	switch (m_Type)
	{
		case MaterialType::Dielectric: return (u_Dielectric)->Scatter(ray, rec, attenuation, scattered, local_rand_state);
		case MaterialType::Lambert: return u_Lambert->Scatter(ray, rec, attenuation, scattered, local_rand_state);
		case MaterialType::Metal: return u_Metal->Scatter(ray, rec, attenuation, scattered, local_rand_state);
	}
	assert(false);
	return false;
}
