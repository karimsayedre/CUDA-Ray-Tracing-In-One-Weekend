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

	// Note: Could use this as a class with member functions but NVCC wouldn't show them in PTXAS info
	struct Materials
	{
		Vec3*  Albedo;
		float* Fuzz;
		float* Ior;
		Vec3*  Flags;

		uint32_t Count = 0;
	};

	template<ExecutionMode Mode>
	__host__ inline Materials* Init(const uint32_t capacity)
	{
		Materials h_materials;
		h_materials.Albedo = MemPolicy<Mode>::template Alloc<Vec3>(capacity);
		h_materials.Flags  = MemPolicy<Mode>::template Alloc<Vec3>(capacity);
		h_materials.Fuzz   = MemPolicy<Mode>::template Alloc<float>(capacity);
		h_materials.Ior	   = MemPolicy<Mode>::template Alloc<float>(capacity);

		Materials* d_Materials = MemPolicy<Mode>::template Alloc<Materials>(1);

		if constexpr (Mode == ExecutionMode::GPU)
			CHECK_CUDA_ERRORS(cudaMemcpy(d_Materials, &h_materials, sizeof(Materials), cudaMemcpyHostToDevice));
		else
			*d_Materials = h_materials;

		return d_Materials;
	}

	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__device__ __host__ CPU_ONLY_INLINE void Add(const MaterialType type, const Vec3& albedo, const float fuzz = 0.0f, const float ior = 1.0f)
	{
		const RenderParams* params = GetParams();

		params->Materials->Albedo[params->Materials->Count] = albedo;
		params->Materials->Fuzz[params->Materials->Count]	= fuzz;
		params->Materials->Ior[params->Materials->Count]	= ior;

		// Set material flags based on type
		switch (type)
		{
			case MaterialType::Lambert:
				params->Materials->Flags[params->Materials->Count].x = 1;
				params->Materials->Flags[params->Materials->Count].y = 0;
				params->Materials->Flags[params->Materials->Count].z = 0;
				break;
			case MaterialType::Metal:
				params->Materials->Flags[params->Materials->Count].x = 0;
				params->Materials->Flags[params->Materials->Count].y = 1;
				params->Materials->Flags[params->Materials->Count].z = 0;
				break;
			case MaterialType::Dielectric:
				params->Materials->Flags[params->Materials->Count].x = 0;
				params->Materials->Flags[params->Materials->Count].y = 0;
				params->Materials->Flags[params->Materials->Count].z = 1;
				break;
		}
		params->Materials->Count++;
	}

	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__device__ __host__ CPU_ONLY_INLINE bool Scatter(Ray& incomingRay, const HitRecord& rec, Vec3& currAttenuation, uint32_t& randSeed)
	{
		const RenderParams* params = GetParams();

		// 1. Get one random vector for all calculations
		const Vec3 randomVec = RandomVec3(randSeed);
		currAttenuation *= params->Materials->Albedo[rec.PrimitiveIndex];

		// 2. Lambert direction - immediate computation
		Vec3		lambertDir	  = rec.Normal + randomVec;
		const float lambertDirLen = glm::length(lambertDir);
		lambertDir				  = lambertDirLen > 0.001f ? lambertDir / lambertDirLen : rec.Normal;

		// 3. Metal direction - immediate computation
		const Vec3 reflected = Reflect(incomingRay.Direction, rec.Normal);
		const Vec3 metalDir	 = reflected + randomVec * params->Materials->Fuzz[rec.PrimitiveIndex];

		// 4. Dielectric direction - computation with reused variables
		Vec3 dielectricDir;
		{
			const float ior			  = params->Materials->Ior[rec.PrimitiveIndex];
			const float dotRayNormal  = dot(incomingRay.Direction, rec.Normal);
			const bool	frontFace	  = dotRayNormal < 0.0f;
			const Vec3& outwardNormal = frontFace ? rec.Normal : -rec.Normal;
			const float niOverNt	  = frontFace ? (1.0f / ior) : ior;
			const float cosine		  = frontFace ? -dotRayNormal : dotRayNormal;

			Vec3		refracted;
			const bool	canRefract	 = Refract(incomingRay.Direction, outwardNormal, niOverNt, refracted);
			const float reflectProb	 = canRefract ? Reflectance(cosine, ior) : 1.0f;
			const Vec3	reflectedDir = Reflect(incomingRay.Direction, outwardNormal);
			dielectricDir			 = RandomFloat(randSeed) < reflectProb ? reflectedDir : refracted;
		}

		// 5. Material weights - compute once
		const Vec3	flags		   = params->Materials->Flags[rec.PrimitiveIndex];
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
