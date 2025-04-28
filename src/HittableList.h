#pragma once

#include "AABB.h"
#include "CPU_GPU.h"
#include "Renderer.h"

namespace Hitables
{

	struct PrimitiveList
	{
		Vec3*	 Centers;
		float*	 Radii;
		AABB*	 AABBs;
		uint16_t Count = 0;
	};

	template<ExecutionMode M>
	__host__ inline PrimitiveList* Init(const uint32_t capacity)
	{
		PrimitiveList h_HittableList;
		h_HittableList.Centers = MemPolicy<M>::template Alloc<Vec3>(capacity);
		h_HittableList.Radii   = MemPolicy<M>::template Alloc<float>(capacity);
		h_HittableList.AABBs   = MemPolicy<M>::template Alloc<AABB>(capacity);

		PrimitiveList* d_HittableList = MemPolicy<M>::template Alloc<PrimitiveList>(1);

		if constexpr (M == ExecutionMode::GPU)
		{
			// device allocation + copy
			CHECK_CUDA_ERRORS(cudaMemcpy(d_HittableList, &h_HittableList, sizeof(PrimitiveList), cudaMemcpyHostToDevice));
		}
		else
		{
			// plain host allocation + copy
			*d_HittableList = h_HittableList;
		}

		return d_HittableList;
	}

	__device__ __host__ inline void Add(const Vec3& position, const float radius)
	{
		const auto* params = GetParams();

		params->List->Centers[params->List->Count] = position;
		params->List->Radii[params->List->Count]   = radius;
		params->List->AABBs[params->List->Count]   = AABB(position - radius, position + radius);
		params->List->Count++;
	}

	__device__ __host__ inline bool IntersectPrimitive(const Ray& ray, const float tMin, float& tMax, HitRecord& record, const uint16_t sphereIndex)
	{
		const auto* params		 = GetParams();
		const Vec3	oc			 = ray.Origin - params->List->Centers[sphereIndex];
		const float a			 = glm::dot(ray.Direction, ray.Direction);
		const float b			 = glm::dot(oc, ray.Direction);
		const float c			 = glm::dot(oc, oc) - params->List->Radii[sphereIndex] * params->List->Radii[sphereIndex];
		const float discriminant = b * b - a * c;

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
		record.Location		  = ray.PointAtT(t);
		record.Normal		  = (record.Location - params->List->Centers[sphereIndex]) * (1.0f / params->List->Radii[sphereIndex]); // Normalize using multiplication
		record.PrimitiveIndex = sphereIndex;

		tMax = t; // Update tMax for the next intersection test
		return true;
	}

} // namespace Hitables
