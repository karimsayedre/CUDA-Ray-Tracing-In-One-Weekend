#pragma once

#include "AABB.h"
#include "CPU_GPU.h"
#include "Renderer.h"

namespace Hitables
{

	// Note: Could use this as a class with member functions but NVCC wouldn't show them in PTXAS info
	struct PrimitiveList
	{
		Vec3* __restrict__ Centers;
		float* __restrict__ Radii;
		AABB* __restrict__ AABBs;
		uint32_t Count = 0;
	};

	template<ExecutionMode Mode>
	__host__ inline PrimitiveList* Init(const uint32_t capacity)
	{
		PrimitiveList h_HittableList;
		h_HittableList.Centers = MemPolicy<Mode>::template Alloc<Vec3>(capacity);
		h_HittableList.Radii   = MemPolicy<Mode>::template Alloc<float>(capacity);
		h_HittableList.AABBs   = MemPolicy<Mode>::template Alloc<AABB>(capacity);

		PrimitiveList* d_HittableList = MemPolicy<Mode>::template Alloc<PrimitiveList>(1);

		if constexpr (Mode == ExecutionMode::GPU)
			CHECK_CUDA_ERRORS(cudaMemcpy(d_HittableList, &h_HittableList, sizeof(PrimitiveList), cudaMemcpyHostToDevice));
		else
			*d_HittableList = h_HittableList;

		return d_HittableList;
	}

	__device__ __host__ CPU_ONLY_INLINE void Add(const Vec3& position, const float radius)
	{
		const RenderParams* __restrict__ params = GetParams();

		params->List->Centers[params->List->Count] = position;
		params->List->Radii[params->List->Count]   = radius;
		params->List->AABBs[params->List->Count]   = AABB(position - radius, position + radius);
		params->List->Count++;
	}

	__device__ __host__ CPU_ONLY_INLINE bool IntersectPrimitive(const Ray& ray, const float tMin, float& tMax, HitRecord& record, const uint32_t sphereIndex)
	{
		const RenderParams* __restrict__ params = GetParams();
		const Vec3	oc							= ray.Origin - params->List->Centers[sphereIndex];
		const float a							= glm::dot(ray.Direction, ray.Direction);
		const float b							= glm::dot(oc, ray.Direction);
		const float c							= glm::dot(oc, oc) - params->List->Radii[sphereIndex] * params->List->Radii[sphereIndex];
		const float discriminant				= b * b - a * c;

		if (discriminant <= 0.0f)
			return false; // No intersection

		const float sqrtD = glm::sqrt(discriminant);
		const float t0	  = (-b - sqrtD) / a;
		const float t1	  = (-b + sqrtD) / a;

		// Pick the closest valid t
		const float t = (t0 > tMin && t0 < tMax) ? t0 : ((t1 > tMin && t1 < tMax) ? t1 : -1.0f);
		if (t < 0.0f)
			return false;

		// Compute intersection data
		record.Location = ray.PointAtT(t);

		const float invRadius = 1.0f / params->List->Radii[sphereIndex];
		const Vec3	negCenter = -params->List->Centers[sphereIndex];
#ifdef __CUDA_ARCH__
		record.Normal.x = fmaf(record.Location.x, invRadius, negCenter.x * invRadius);
		record.Normal.y = fmaf(record.Location.y, invRadius, negCenter.y * invRadius);
		record.Normal.z = fmaf(record.Location.z, invRadius, negCenter.z * invRadius);
#else
		record.Normal = (record.Location + negCenter) * invRadius; // Normalize using multiplication
#endif
		record.PrimitiveIndex = sphereIndex;

		tMax = t; // Update tMax for the next intersection test
		return true;
	}

} // namespace Hitables
