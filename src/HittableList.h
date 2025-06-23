#pragma once

#include "CPU_GPU.h"
#include "Renderer.h"
#include "AABB.h"

namespace Hitables
{

	// Note: Could use this as a class with member functions but NVCC wouldn't show them in PTXAS info
	struct PrimitiveList
	{
		Vec4* __restrict__ CentersAndRadius;
		AABB* __restrict__ AABBs;
		uint32_t Count;
	};

	template<ExecutionMode Mode>
	[[nodiscard]] __host__ inline PrimitiveList* Init(const uint32_t capacity)
	{
		PrimitiveList hittableList;
		hittableList.CentersAndRadius = MemPolicy<Mode>::template Alloc<Vec4>(capacity);
		hittableList.AABBs			  = MemPolicy<Mode>::template Alloc<AABB>(capacity);
		hittableList.Count			  = 0;
		PrimitiveList* d_HittableList = MemPolicy<Mode>::template Alloc<PrimitiveList>(1);

		if constexpr (Mode == ExecutionMode::GPU)
			CHECK_CUDA_ERRORS(cudaMemcpy(d_HittableList, &hittableList, sizeof(PrimitiveList), cudaMemcpyHostToDevice));
		else
			*d_HittableList = hittableList;

		return d_HittableList;
	}

	template<ExecutionMode Mode>
	__host__ inline void Destroy(PrimitiveList* list)
	{
		if constexpr (Mode == ExecutionMode::GPU)
		{
			PrimitiveList hostList;
			CHECK_CUDA_ERRORS(cudaMemcpy(&hostList, list, sizeof(PrimitiveList), cudaMemcpyDeviceToHost));

			MemPolicy<Mode>::Free(hostList.CentersAndRadius);
			MemPolicy<Mode>::Free(hostList.AABBs);
			MemPolicy<Mode>::Free(list);
		}
		else
		{
			MemPolicy<Mode>::Free(list->CentersAndRadius);
			MemPolicy<Mode>::Free(list->AABBs);
			MemPolicy<Mode>::Free(list);
		}
	}

	__device__ __host__ CPU_ONLY_INLINE void Add(const Vec3& position, const float radius)
	{
		const RenderParams* __restrict__ params = GetParams();

		params->List->CentersAndRadius[params->List->Count] = Vec4(position, radius);
		params->List->AABBs[params->List->Count]			= AABB(position - radius, position + radius);
		params->List->Count++;
	}

	[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE bool IntersectPrimitive(const Ray& ray, const float tMin, float& tMax, HitRecord& record, const uint32_t sphereIndex)
	{
		const RenderParams* params	= GetParams();
		const auto [center, radius] = params->List->CentersAndRadius[sphereIndex];
		const Vec3	oc				= ray.Origin - center;
		const float a				= glm::dot(ray.Direction, ray.Direction);
		const float b				= glm::dot(oc, ray.Direction);
		const float c				= glm::dot(oc, oc) - radius * radius;
		const float discriminant	= b * b - a * c;

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

		const float invRadius = 1.0f / radius;
		const Vec3	negCenter = -center;
#ifdef __CUDA_ARCH__
		record.Normal.x = std::fmaf(record.Location.x, invRadius, negCenter.x * invRadius);
		record.Normal.y = std::fmaf(record.Location.y, invRadius, negCenter.y * invRadius);
		record.Normal.z = std::fmaf(record.Location.z, invRadius, negCenter.z * invRadius);
#else
		record.Normal = (record.Location + negCenter) * invRadius; // Normalize using multiplication
#endif
		record.PrimitiveIndex = sphereIndex;

		tMax = t; // Update tMax for the next intersection test
		return true;
	}

} // namespace Hitables
