#pragma once
#include <numbers>

#include "pch.h"
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
		Vec3* __restrict__ Albedo;
		float* __restrict__ Fuzz;
		float* __restrict__ Ior;
		Vec3* __restrict__ Flags;

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

	__device__ __host__ CPU_ONLY_INLINE void Add(const MaterialType type, const Vec3& albedo, const float fuzz = 0.0f, const float ior = 1.0f)
	{
		const RenderParams* __restrict__ params = GetParams();

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

	__device__ __host__ CPU_ONLY_INLINE bool Scatter(Ray& ray, const HitRecord& rec, Vec3& attenuation, uint32_t& randSeed)
	{
		const RenderParams* __restrict__ params = GetParams();
		const auto& m							= params->Materials;

		Vec3  albedo = m->Albedo[rec.PrimitiveIndex];
		float fuzz	 = m->Fuzz[rec.PrimitiveIndex];
		float ior	 = m->Ior[rec.PrimitiveIndex];
		Vec3  w		 = m->Flags[rec.PrimitiveIndex];

		// normalize weights
		float sumW	= w.x + w.y + w.z + 1e-6f;
		Vec3  normW = w / sumW;

		const Vec3 rand3 = RandomVec3(randSeed);

		// Precompute directions
		const Vec3& unitDir	   = ray.Direction; // already normalized
		Vec3		lambertDir = normalize(rec.Normal + rand3);
		Vec3		metalRef   = reflect(unitDir, rec.Normal);
		Vec3		metalDir   = metalRef + fuzz * rand3;

		// Dielectric components using faceNormal
		float frontFaceMask = float(dot(unitDir, rec.Normal) < 0.0f);
		Vec3  faceNormal	= frontFaceMask * rec.Normal + (1.0f - frontFaceMask) * -rec.Normal;
		float cosTheta		= fminf(dot(-unitDir, faceNormal), 1.0f);
		float sinTheta		= sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
		float etaiOverEtat	= frontFaceMask * (1.0f / ior) + (1.0f - frontFaceMask) * ior;
		float cannotRefract = float(etaiOverEtat * sinTheta > 1.0f);
		float reflectProb	= cannotRefract + (1.0f - cannotRefract) * Schlick(cosTheta, ior);
		float rnd			= RandomFloat(randSeed);
		float isReflect		= float(rnd < reflectProb);

		Vec3 refracted = RefractBranchless(unitDir, faceNormal, etaiOverEtat);
		Vec3 dielecDir = isReflect * reflect(unitDir, faceNormal) + (1.0f - isReflect) * refracted;

		// Composite direction and normalize
		Vec3 dir = lambertDir * normW.x + metalDir * normW.y + dielecDir * normW.z;
		ray		 = Ray(rec.Location, normalize(dir));

		// Branchless attenuation: lambert & metal albedo, dielectric = 1
		Vec3 att = albedo * (normW.x + normW.y) + Vec3(1.0f) * normW.z;
		attenuation *= att * sumW;
#if 1
		const float scatterDot = dot(dir, rec.Normal);
		return (scatterDot > 0.0f) || (w.z > 0.0f);

#endif
	}

} // namespace Mat
