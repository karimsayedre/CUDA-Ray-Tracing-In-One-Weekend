#pragma once
#include "Random.h"
#include "Ray.h"
#include "Vec3.h"

enum class MaterialType
{
	Lambert,
	Metal,
	Dielectric
};

struct Materials
{
	// namespace MaterialSoA
	Vec3*	  Albedo;
	Float*	  Fuzz;
	Float*	  Ior;
	uint32_t* MaterialFlagsX;
	uint32_t* MaterialFlagsY;
	uint32_t* MaterialFlagsZ;

	uint32_t Count = 0;

	__host__ inline static void Init(Materials*& d_materials, uint32_t maxMaterialCount)
	{
		Materials h_materials;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Albedo, maxMaterialCount * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Fuzz, maxMaterialCount * sizeof(Float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Ior, maxMaterialCount * sizeof(Float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.MaterialFlagsX, maxMaterialCount * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.MaterialFlagsY, maxMaterialCount * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.MaterialFlagsZ, maxMaterialCount * sizeof(uint32_t)));

		// Allocate memory for BVH structure on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&d_materials, sizeof(Materials)));

		// Copy initialized BVH data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_materials, &h_materials, sizeof(Materials), cudaMemcpyHostToDevice));
	}

	__device__ void Add(MaterialType type, const Vec3& albedo, Float fuzz = 0.0f, Float ior = 1.0f)
	{
		Albedo[Count] = albedo;
		Fuzz[Count]	  = fuzz;
		Ior[Count]	  = ior;

		// Set material flags based on type
		switch (type)
		{
			case MaterialType::Lambert:
				MaterialFlagsX[Count] = 1;
				MaterialFlagsY[Count] = 0;
				MaterialFlagsZ[Count] = 0;
				break;
			case MaterialType::Metal:
				MaterialFlagsX[Count] = 0;
				MaterialFlagsY[Count] = 1;
				MaterialFlagsZ[Count] = 0;
				break;
			case MaterialType::Dielectric:
				MaterialFlagsX[Count] = 0;
				MaterialFlagsY[Count] = 0;
				MaterialFlagsZ[Count] = 1;
				break;
		}
		Count++;
	}

	__device__ bool Scatter(Ray& incomingRay, Ray& scatteredRay, const HitRecord& rec, Vec3& attenuation, uint32_t& randSeed) const
	{
		// Normalize direction (branchless)
		const Vec3	rayDir		   = incomingRay.Direction();
		const Float rayDirLen	   = fastLength(rayDir);
		const Float rayDirLenRecip = rayDirLen > __float2half(0.0001f) ? __float2half(1.0f) / rayDirLen : __float2half(1.0f);
		const Vec3	unitDir		   = rayDir * rayDirLenRecip;

		// Normal based on surface face
		const Float dotRayNormal  = dot(unitDir, rec.Normal);
		const bool	frontFace	  = dotRayNormal < __float2half(0.0f);
		const Vec3	outwardNormal = frontFace ? rec.Normal : -rec.Normal;

		// Calculate scattered directions for all types
		const Vec3 randomVec = RandomVec3(randSeed);

		// Lambert scattering direction
		Vec3		lambertDir	  = rec.Normal + randomVec;
		const Float lambertDirLen = glm::length(lambertDir);
		lambertDir				  = lambertDirLen > __float2half(0.001f) ? lambertDir / lambertDirLen : rec.Normal;

		// Metal reflection direction
		const Vec3 reflected = reflect(unitDir, rec.Normal);
		const Vec3 metalDir	 = reflected + randomVec * Fuzz[rec.MaterialIndex];

		// Dielectric refraction/reflection
		Float		ior		 = Ior[rec.MaterialIndex];
		const Float niOverNt = frontFace ? (__float2half(1.0f) / ior) : ior;
		const Float cosine	 = frontFace ? -dotRayNormal : dotRayNormal;

		Vec3		refracted;
		const bool	canRefract	  = refract(unitDir, outwardNormal, niOverNt, refracted);
		const Float reflectProb	  = canRefract ? Reflectance(cosine, ior) : 1.0f;
		const Float randVal		  = RandomFloat(randSeed);
		const Vec3	dielectricDir = (randVal < reflectProb) ? reflect(unitDir, outwardNormal) : refracted;

		// Selection of direction based on material parameters (branchless)
		// m_MaterialFlags: [0] = Lambert component, [1] = Metal component, [2] = Dielectric component
		// Values between 0 and 1 allow for blended materials
		const Float lambertWeight	 = __float2half((Float)MaterialFlagsX[rec.MaterialIndex]);
		const Float metalWeight		 = __float2half((Float)MaterialFlagsY[rec.MaterialIndex]);
		const Float dielectricWeight = __float2half((Float)MaterialFlagsZ[rec.MaterialIndex]);

		// Normalize weights to sum to 1.0
		const Float totalWeight		 = lambertWeight + metalWeight + dielectricWeight;
		const Float weightNormalizer = totalWeight > __float2half(0.0001f) ? __float2half(1.0f) / totalWeight : __float2half(0.0f);

		const Float normLambertWeight	 = lambertWeight * weightNormalizer;
		const Float normMetalWeight		 = metalWeight * weightNormalizer;
		const Float normDielectricWeight = dielectricWeight * weightNormalizer;

		// Blend directions
		Vec3 finalDir =
			lambertDir * normLambertWeight + metalDir * normMetalWeight + dielectricDir * normDielectricWeight;

		// Ensure direction is normalized
		finalDir = glm::normalize(finalDir);

		// Calculate if scattering is valid (mainly for metal)
		const Float scatterDot	 = dot(finalDir, rec.Normal);
		const bool	validScatter = (scatterDot > __float2half(0.0f)) || (dielectricWeight > __float2half(0.0f));

		// Apply attenuation based on material properties
		attenuation *= Albedo[rec.MaterialIndex];

		// Set scattered ray
		scatteredRay = Ray(rec.Location, finalDir);

		return validScatter;
	}
};