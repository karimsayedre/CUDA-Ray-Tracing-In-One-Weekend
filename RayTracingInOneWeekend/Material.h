#pragma once
#include "Hittable.h"
#include "Random.h"
#include "Ray.h"
#include "Vec3.h"

enum class MaterialType
{
	Lambert,
	Metal,
	Dielectric
};

namespace MaterialSoA
{
	// namespace MaterialSoA
	extern __device__ glm::vec3* Albedo;
	extern __device__ float*		Fuzz;
	extern __device__ float*		Ior;
	extern __device__ uint32_t*	MaterialFlagsX;
	extern __device__ uint32_t*	MaterialFlagsY;
	extern __device__ uint32_t*	MaterialFlagsZ;

	__host__ inline static void Init(uint32_t maxMaterialCount)
	{
		// Allocate device memory
		glm::vec3* d_albedo;
		float*	   d_fuzz;
		float*	   d_ior;
		uint32_t*  d_flagsX;
		uint32_t*  d_flagsY;
		uint32_t*  d_flagsZ;

		CHECK_CUDA_ERRORS(cudaMalloc(&d_albedo, maxMaterialCount * sizeof(glm::vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&d_fuzz, maxMaterialCount * sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&d_ior, maxMaterialCount * sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&d_flagsX, maxMaterialCount * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&d_flagsY, maxMaterialCount * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&d_flagsZ, maxMaterialCount * sizeof(uint32_t)));

		// Copy device pointers to __device__ variables
		CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(MaterialSoA::Albedo, &d_albedo, sizeof(glm::vec3*)));
		CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(MaterialSoA::Fuzz, &d_fuzz, sizeof(float*)));
		CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(MaterialSoA::Ior, &d_ior, sizeof(float*)));
		CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(MaterialSoA::MaterialFlagsX, &d_flagsX, sizeof(uint32_t*)));
		CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(MaterialSoA::MaterialFlagsY, &d_flagsY, sizeof(uint32_t*)));
		CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(MaterialSoA::MaterialFlagsZ, &d_flagsZ, sizeof(uint32_t*)));
	}
} // namespace MaterialSoA

struct Materials
{
	__device__ static void Add(
		MaterialType	 type,
		const glm::vec3& albedo,
		float			 fuzz  = 0.0f,
		float			 ior   = 1.0f,
		int				 index = -1)
	{
		if (index == -1)
			return; // Safety check
		MaterialSoA::Albedo[index] = albedo;
		MaterialSoA::Fuzz[index] = fuzz;
		MaterialSoA::Ior[index]	 = ior;

		// Set material flags based on type
		switch (type)
		{
			case MaterialType::Lambert:
				MaterialSoA::MaterialFlagsX[index] = 1;
				MaterialSoA::MaterialFlagsY[index] = 0;
				MaterialSoA::MaterialFlagsZ[index] = 0;
				break;
			case MaterialType::Metal:
				MaterialSoA::MaterialFlagsX[index] = 0;
				MaterialSoA::MaterialFlagsY[index] = 1;
				MaterialSoA::MaterialFlagsZ[index] = 0;
				break;
			case MaterialType::Dielectric:
				MaterialSoA::MaterialFlagsX[index] = 0;
				MaterialSoA::MaterialFlagsY[index] = 0;
				MaterialSoA::MaterialFlagsZ[index] = 1;
				break;
		}
	}

};

class Material
{
  public:
	__device__ static bool Scatter(Ray& incomingRay, Ray& scatteredRay, const HitRecord& rec, glm::vec3& attenuation, uint32_t& randSeed, uint32_t materialIndex);
};


__device__ inline bool Material::Scatter(Ray& incomingRay, Ray& scatteredRay, const HitRecord& rec, glm::vec3& attenuation, uint32_t& randSeed, uint32_t materialIndex)
{
	// Normalize direction (branchless)
	const glm::vec3 rayDir		   = incomingRay.Direction();
	const float		rayDirLen	   = fastLength(rayDir);
	const float		rayDirLenRecip = rayDirLen > 0.0001f ? 1.0f / rayDirLen : 1.0f;
	const glm::vec3 unitDir		   = rayDir * rayDirLenRecip;

	// Normal based on surface face
	const float		dotRayNormal  = dot(unitDir, rec.Normal);
	const bool		frontFace	  = dotRayNormal < 0.0f;
	const glm::vec3 outwardNormal = frontFace ? rec.Normal : -rec.Normal;

	// Calculate scattered directions for all types
	const glm::vec3 randomVec = RandomVec3(randSeed);

	// Lambert scattering direction
	glm::vec3	lambertDir	  = rec.Normal + randomVec;
	const float lambertDirLen = glm::length(lambertDir);
	lambertDir				  = lambertDirLen > 0.001f ? lambertDir / lambertDirLen : rec.Normal;

	// Metal reflection direction
	const glm::vec3 reflected = reflect(unitDir, rec.Normal);
	const glm::vec3 metalDir  = reflected + randomVec * MaterialSoA::Fuzz[materialIndex];

	// Dielectric refraction/reflection
	float		ior		 = MaterialSoA::Ior[materialIndex];
	const float niOverNt = frontFace ? (1.0f / ior) : ior;
	const float cosine	 = frontFace ? -dotRayNormal : dotRayNormal;

	glm::vec3		refracted;
	const bool		canRefract	  = refract(unitDir, outwardNormal, niOverNt, refracted);
	const float		reflectProb	  = canRefract ? Reflectance(cosine, ior) : 1.0f;
	const float		randVal		  = RandomFloat(randSeed);
	const glm::vec3 dielectricDir = (randVal < reflectProb) ? reflect(unitDir, outwardNormal) : refracted;

	// Selection of direction based on material parameters (branchless)
	// m_MaterialFlags: [0] = Lambert component, [1] = Metal component, [2] = Dielectric component
	// Values between 0 and 1 allow for blended materials
	const float lambertWeight	 = (float)MaterialSoA::MaterialFlagsX[materialIndex];
	const float metalWeight		 = (float)MaterialSoA::MaterialFlagsY[materialIndex];
	const float dielectricWeight = (float)MaterialSoA::MaterialFlagsZ[materialIndex];

	// Normalize weights to sum to 1.0
	const float totalWeight		 = lambertWeight + metalWeight + dielectricWeight;
	const float weightNormalizer = totalWeight > 0.0001f ? 1.0f / totalWeight : 0.0f;

	const float normLambertWeight	 = lambertWeight * weightNormalizer;
	const float normMetalWeight		 = metalWeight * weightNormalizer;
	const float normDielectricWeight = dielectricWeight * weightNormalizer;

	// Blend directions
	glm::vec3 finalDir =
		lambertDir * normLambertWeight + metalDir * normMetalWeight + dielectricDir * normDielectricWeight;

	// Ensure direction is normalized
	finalDir = glm::normalize(finalDir);

	// Calculate if scattering is valid (mainly for metal)
	const float scatterDot	 = dot(finalDir, rec.Normal);
	const bool	validScatter = (scatterDot > 0.0f) || (dielectricWeight > 0.0f);

	// Apply attenuation based on material properties
	attenuation *= MaterialSoA::Albedo[materialIndex];

	// Set scattered ray
	scatteredRay = Ray(rec.Location, finalDir);

	return validScatter;
}
