#pragma once
#include "Random.h"
#include "Vec3.h"

namespace Mat
{
	enum class MaterialType : uint8_t
	{
		Lambert,
		Metal,
		Dielectric
	};

	struct Materials
	{
		Vec3*  Albedo;
		float* Fuzz;
		float* Ior;
		Vec3*  Flags;

		uint32_t Count = 0;
	};

	__host__ static void InitMaterials(Materials*& d_Materials, const uint32_t maxMaterialCount)
	{
		Materials h_materials;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Albedo, maxMaterialCount * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Flags, maxMaterialCount * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Fuzz, maxMaterialCount * sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Ior, maxMaterialCount * sizeof(float)));

		// Allocate memory for BVH structure on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&d_Materials, sizeof(Materials)));

		// Copy initialized BVH data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_Materials, &h_materials, sizeof(Materials), cudaMemcpyHostToDevice));
	}

	__device__ void Add(Materials* mats, const MaterialType type, const Vec3& albedo, const float fuzz = 0.0f, const float ior = 1.0f)
	{
		mats->Albedo[mats->Count] = albedo;
		mats->Fuzz[mats->Count]	  = fuzz;
		mats->Ior[mats->Count]	  = ior;

		// Set material flags based on type
		switch (type)
		{
			case MaterialType::Lambert:
				mats->Flags[mats->Count].x = 1;
				mats->Flags[mats->Count].y = 0;
				mats->Flags[mats->Count].z = 0;
				break;
			case MaterialType::Metal:
				mats->Flags[mats->Count].x = 0;
				mats->Flags[mats->Count].y = 1;
				mats->Flags[mats->Count].z = 0;
				break;
			case MaterialType::Dielectric:
				mats->Flags[mats->Count].x = 0;
				mats->Flags[mats->Count].y = 0;
				mats->Flags[mats->Count].z = 1;
				break;
		}
		mats->Count++;
	}

	__device__ __forceinline__ bool Scatter(Materials* mats, Ray& incomingRay, const HitRecord& rec, Vec3& currAttenuation, uint32_t& randSeed)
	{
		// 1. Compute normalized direction with minimal temporaries
		const Vec3& rayDir = incomingRay.Direction();

		// 2. Get one random vector for all calculations
		const Vec3 randomVec = RandomVec3(randSeed);

		// 3. Lambert direction - immediate computation
		Vec3		lambertDir	  = rec.Normal + randomVec;
		const float lambertDirLen = glm::length(lambertDir);
		lambertDir				  = lambertDirLen > 0.001f ? lambertDir / lambertDirLen : rec.Normal;

		// 4. Metal direction - immediate computation
		const Vec3 reflected = Reflect(rayDir, rec.Normal);
		const Vec3 metalDir	 = reflected + randomVec * mats->Fuzz[rec.PrimitiveIndex];

		// 5. Dielectric direction - computation with reused variables
		Vec3 dielectricDir;
		{
			const float ior			  = mats->Ior[rec.PrimitiveIndex];
			const float dotRayNormal  = dot(rayDir, rec.Normal);
			const bool	frontFace	  = dotRayNormal < 0.0f;
			const Vec3& outwardNormal = frontFace ? rec.Normal : -rec.Normal;
			const float niOverNt	  = frontFace ? (1.0f / ior) : ior;
			const float cosine		  = frontFace ? -dotRayNormal : dotRayNormal;

			Vec3		refracted;
			const bool	canRefract	 = Refract(rayDir, outwardNormal, niOverNt, refracted);
			const float reflectProb	 = canRefract ? Reflectance(cosine, ior) : 1.0f;
			const Vec3	reflectedDir = Reflect(rayDir, outwardNormal);
			dielectricDir			 = (RandomFloat(randSeed) < reflectProb) ? reflectedDir : refracted;
		}

		// 6. Material weights - compute once
		const Vec3	flags		   = mats->Flags[rec.PrimitiveIndex];
		const float totalWeight	   = flags.x + flags.y + flags.z;
		const float invTotalWeight = totalWeight > 0.0001f ? 1.0f / totalWeight : 0.0f;

		// 7. Blend directions - direct computation with minimal temporaries
		const Vec3 finalDir = lambertDir * (flags.x * invTotalWeight) + metalDir * (flags.y * invTotalWeight) + dielectricDir * (flags.z * invTotalWeight);

		// 8. Check valid scatter
		const float scatterDot	 = dot(finalDir, rec.Normal);
		const bool	validScatter = (scatterDot > 0.0f) || (flags.z > 0.0f);

		// 9. Apply attenuation and update ray
		currAttenuation *= mats->Albedo[rec.PrimitiveIndex];
		incomingRay = {rec.Location, finalDir};

		return validScatter;
	}

} // namespace Mat
