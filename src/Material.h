#pragma once
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
		Vec4* __restrict__ AlbedoIOR; // x: XYZ, y: XYZ, z: XYZ, w: W
		Vec4* __restrict__ FlagsFuzz; // x: Lambert, y: Metal, z: Dielectric, w: Fuzz

		uint32_t Count;
	};

	template<ExecutionMode Mode>
	[[nodiscard]] __host__ inline Materials* Init(const uint32_t capacity)
	{
		Materials materials;
		materials.AlbedoIOR	   = MemPolicy<Mode>::template Alloc<Vec4>(capacity);
		materials.FlagsFuzz	   = MemPolicy<Mode>::template Alloc<Vec4>(capacity);
		materials.Count		   = 0;
		Materials* d_Materials = MemPolicy<Mode>::template Alloc<Materials>(1);

		if constexpr (Mode == ExecutionMode::GPU)
			CHECK_CUDA_ERRORS(cudaMemcpy(d_Materials, &materials, sizeof(Materials), cudaMemcpyHostToDevice));
		else
			*d_Materials = materials;

		return d_Materials;
	}

	__device__ __host__ CPU_ONLY_INLINE void Add(const MaterialType type, const Vec3& albedo, const float fuzz = 0.0f, const float ior = 1.0f)
	{
		const RenderParams* __restrict__ params					 = GetParams();
		params->Materials->AlbedoIOR[params->Materials->Count]	 = Vec4(albedo, ior);
		params->Materials->FlagsFuzz[params->Materials->Count].W = fuzz;

		// Set material flags based on type
		switch (type)
		{
			case MaterialType::Lambert:
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.x = 1;
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.y = 0;
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.z = 0;
				break;
			case MaterialType::Metal:
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.x = 0;
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.y = 1;
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.z = 0;
				break;
			case MaterialType::Dielectric:
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.x = 0;
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.y = 0;
				params->Materials->FlagsFuzz[params->Materials->Count].XYZ.z = 1;
				break;
		}
		params->Materials->Count++;
	}

	[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE bool Scatter(Ray& ray, const HitRecord& rec, Vec3& attenuation, uint32_t& randSeed)
	{
		const RenderParams* params = GetParams();

		Vec4 albedoIOR	= params->Materials->AlbedoIOR[rec.PrimitiveIndex];
		Vec4 weightFuzz = params->Materials->FlagsFuzz[rec.PrimitiveIndex];

		// normalize weights
		float sumW	= weightFuzz.XYZ.x + weightFuzz.XYZ.y + weightFuzz.XYZ.z + 1e-6f;
		Vec3  normW = weightFuzz.XYZ / sumW;

		const Vec3 rand3 = RandomVec3(randSeed);

		// Precompute directions
		const Vec3& unitDir	   = ray.Direction; // already normalized
		Vec3		lambertDir = glm::normalize(rec.Normal + rand3);
		Vec3		metalRef   = reflect(unitDir, rec.Normal);
		Vec3		metalDir   = glm::normalize(metalRef + weightFuzz.W * rand3);

		// Dielectric components using faceNormal
		float frontFaceMask = float(dot(unitDir, rec.Normal) < 0.0f);
		Vec3  faceNormal	= frontFaceMask * rec.Normal + (1.0f - frontFaceMask) * -rec.Normal;
		float cosTheta		= std::fminf(dot(-unitDir, faceNormal), 1.0f);
		float sinTheta		= std::sqrtf(std::fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
		float etaiOverEtat	= frontFaceMask * (1.0f / albedoIOR.W) + (1.0f - frontFaceMask) * albedoIOR.W;
		float cannotRefract = float(etaiOverEtat * sinTheta > 1.0f);
		float reflectProb	= cannotRefract + (1.0f - cannotRefract) * Schlick(cosTheta, albedoIOR.W);
		float isReflect		= float(rand3.x < reflectProb);

		Vec3 refracted = RefractBranchless(unitDir, faceNormal, etaiOverEtat);
		Vec3 dielecDir = isReflect * reflect(unitDir, faceNormal) + (1.0f - isReflect) * refracted;

		// Composite direction and normalize
		Vec3 dir = lambertDir * normW.x + metalDir * normW.y + dielecDir * normW.z;
		ray		 = Ray(rec.Location, normalize(dir));

		// Branchless attenuation: lambert & metal albedo, dielectric = 1
		Vec3 att = reinterpret_cast<Vec3&>(albedoIOR) * (normW.x + normW.y) + Vec3(1.0f) * normW.z;
		attenuation *= att * sumW;

		// Early exit on no scatter
		const float scatterDot = dot(dir, rec.Normal);
		return (scatterDot > 0.0f) || (weightFuzz.XYZ.z > 0.0f);
	}

} // namespace Mat
