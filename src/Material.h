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

	__host__ static Materials* Init(const uint32_t maxMaterialCount)
	{
		Materials h_materials;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Albedo, maxMaterialCount * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Flags, maxMaterialCount * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Fuzz, maxMaterialCount * sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_materials.Ior, maxMaterialCount * sizeof(float)));

		// Allocate memory for BVHNode structure on the device
		Materials* d_Materials;
		CHECK_CUDA_ERRORS(cudaMalloc(&d_Materials, sizeof(Materials)));

		// Copy initialized BVHNode data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_Materials, &h_materials, sizeof(Materials), cudaMemcpyHostToDevice));
		return d_Materials;
	}

	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__device__ void Add(const MaterialType type, const Vec3& albedo, const float fuzz = 0.0f, const float ior = 1.0f)
	{
		d_Params.Materials->Albedo[d_Params.Materials->Count] = albedo;
		d_Params.Materials->Fuzz[d_Params.Materials->Count]	  = fuzz;
		d_Params.Materials->Ior[d_Params.Materials->Count]	  = ior;

		// Set material flags based on type
		switch (type)
		{
			case MaterialType::Lambert:
				d_Params.Materials->Flags[d_Params.Materials->Count].x = 1;
				d_Params.Materials->Flags[d_Params.Materials->Count].y = 0;
				d_Params.Materials->Flags[d_Params.Materials->Count].z = 0;
				break;
			case MaterialType::Metal:
				d_Params.Materials->Flags[d_Params.Materials->Count].x = 0;
				d_Params.Materials->Flags[d_Params.Materials->Count].y = 1;
				d_Params.Materials->Flags[d_Params.Materials->Count].z = 0;
				break;
			case MaterialType::Dielectric:
				d_Params.Materials->Flags[d_Params.Materials->Count].x = 0;
				d_Params.Materials->Flags[d_Params.Materials->Count].y = 0;
				d_Params.Materials->Flags[d_Params.Materials->Count].z = 1;
				break;
		}
		d_Params.Materials->Count++;
	}

	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__device__ bool Scatter(Ray& incomingRay, const HitRecord& rec, Vec3& currAttenuation, uint32_t& randSeed)
	{
		// 1. Get one random vector for all calculations
		const Vec3 randomVec = RandomVec3(randSeed);
		currAttenuation *= d_Params.Materials->Albedo[rec.PrimitiveIndex];

		// 2. Lambert direction - immediate computation
		Vec3		lambertDir	  = rec.Normal + randomVec;
		const float lambertDirLen = glm::length(lambertDir);
		lambertDir				  = lambertDirLen > 0.001f ? lambertDir / lambertDirLen : rec.Normal;

		// 3. Metal direction - immediate computation
		const Vec3 reflected = Reflect(incomingRay.Direction, rec.Normal);
		const Vec3 metalDir	 = reflected + randomVec * d_Params.Materials->Fuzz[rec.PrimitiveIndex];

		// 4. Dielectric direction - computation with reused variables
		Vec3 dielectricDir;
		{
			const float ior			  = d_Params.Materials->Ior[rec.PrimitiveIndex];
			const float dotRayNormal  = dot(incomingRay.Direction, rec.Normal);
			const bool	frontFace	  = dotRayNormal < 0.0f;
			const Vec3& outwardNormal = frontFace ? rec.Normal : -rec.Normal;
			const float niOverNt	  = frontFace ? (1.0f / ior) : ior;
			const float cosine		  = frontFace ? -dotRayNormal : dotRayNormal;

			Vec3		refracted;
			const bool	canRefract	 = Refract(incomingRay.Direction, outwardNormal, niOverNt, refracted);
			const float reflectProb	 = canRefract ? Reflectance(cosine, ior) : 1.0f;
			const Vec3	reflectedDir = Reflect(incomingRay.Direction, outwardNormal);
			dielectricDir			 = (RandomFloat(randSeed) < reflectProb) ? reflectedDir : refracted;
		}

		// 5. Material weights - compute once
		const Vec3	flags		   = d_Params.Materials->Flags[rec.PrimitiveIndex];
		const float totalWeight	   = flags.x + flags.y + flags.z;
		const float invTotalWeight = totalWeight > 0.0001f ? 1.0f / totalWeight : 0.0f;

		// 6. Blend directions - direct computation with minimal temporaries
		const Vec3 finalDir = lambertDir * (flags.x * invTotalWeight) + metalDir * (flags.y * invTotalWeight) + dielectricDir * (flags.z * invTotalWeight);

		// 7. Check valid scatter
		const float scatterDot	 = dot(finalDir, rec.Normal);
		const bool	validScatter = (scatterDot > 0.0f) || (flags.z > 0.0f);

		// 8. Apply attenuation and update ray
		incomingRay = {rec.Location, finalDir};

		return validScatter;
	}

} // namespace Mat
