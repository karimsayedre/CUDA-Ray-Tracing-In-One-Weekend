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
	Vec3*	  Flags;
	//Float* MaterialFlagsX;
	//Float* MaterialFlagsY;
	//Float* MaterialFlagsZ;

	uint32_t Count = 0;

	__host__ inline static void Init(Materials*& d_materials, uint32_t maxMaterialCount)
	{
		Materials h_materials;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Albedo, maxMaterialCount * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Fuzz, maxMaterialCount * sizeof(Float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Ior, maxMaterialCount * sizeof(Float)));
		//CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.MaterialFlagsX, maxMaterialCount * sizeof(Float)));
		//CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.MaterialFlagsY, maxMaterialCount * sizeof(Float)));
		//CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.MaterialFlagsZ, maxMaterialCount * sizeof(Float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Flags, maxMaterialCount * sizeof(Vec3)));

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
				Flags[Count].x = 1;
				Flags[Count].y = 0;
				Flags[Count].z = 0;
				break;
			case MaterialType::Metal:
				Flags[Count].x = 0;
				Flags[Count].y = 1;
				Flags[Count].z = 0;
				break;
			case MaterialType::Dielectric:
				Flags[Count].x = 0;
				Flags[Count].y = 0;
				Flags[Count].z = 1;
				break;
		}
		Count++;
	}

	__device__ __noinline__ bool Scatter(Ray& incomingRay, const HitRecord& rec, Vec3& currAttenuation, uint32_t& randSeed) const
	{
		// 1. Compute normalized direction with minimal temporaries
		const Vec3& rayDir = incomingRay.Direction();

		// 2. Compute normal and face orientation in one step

		// 3. Get one random vector for all calculations
		const Vec3 randomVec = RandomVec3(randSeed);

		// 4. Lambert direction - immediate computation
		Vec3		lambertDir	  = rec.Normal + randomVec;
		const Float lambertDirLen = glm::length(lambertDir);
		lambertDir				  = lambertDirLen > __float2half(0.001f) ? lambertDir / lambertDirLen : rec.Normal;

		// 5. Metal direction - immediate computation
		const Vec3 reflected = reflect(rayDir, rec.Normal);
		const Vec3 metalDir	 = reflected + randomVec * Fuzz[rec.MaterialIndex];

		// 6. Dielectric direction - computation with reused variables
		Vec3 dielectricDir;
		{
			const Float ior			  = Ior[rec.MaterialIndex];
			const Float dotRayNormal  = dot(rayDir, rec.Normal);
			const bool	frontFace	  = dotRayNormal < __float2half(0.0f);
			const Vec3& outwardNormal = frontFace ? rec.Normal : -rec.Normal;
			const Float niOverNt	  = frontFace ? (__float2half(1.0f) / ior) : ior;
			const Float cosine		  = frontFace ? -dotRayNormal : dotRayNormal;

			Vec3		refracted;
			const bool	canRefract	 = refract(rayDir, outwardNormal, niOverNt, refracted);
			const Float reflectProb	 = canRefract ? Reflectance(cosine, ior) : __float2half(1.0f);
			const Vec3	reflectedDir = reflect(rayDir, outwardNormal);
			dielectricDir			 = (RandomFloat(randSeed) < reflectProb) ? reflectedDir : refracted;
		}

		// 7. Material weights - compute once
		const Vec3	flags			 = Flags[rec.MaterialIndex];
		const Float totalWeight	   = flags.x + flags.y + flags.z;
		const Float invTotalWeight	 = totalWeight > __float2half(0.0001f) ? __float2half(1.0f) / totalWeight : __float2half(0.0f);

		// 8. Blend directions - direct computation with minimal temporaries
		const Vec3 finalDir = lambertDir * (flags.x * invTotalWeight) + metalDir * (flags.y * invTotalWeight) + dielectricDir * (flags.z * invTotalWeight);

		// 9. Check valid scatter
		const Float scatterDot	 = dot(finalDir, rec.Normal);
		const bool	validScatter = (scatterDot > __float2half(0.0f)) || (flags.z > __float2half(0.0f));

		// 10. Apply attenuation and update ray (no temporaries)
		currAttenuation *= Albedo[rec.MaterialIndex];
		incomingRay = {rec.Location, finalDir};

		return validScatter;
	}
};