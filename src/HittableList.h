#pragma once

#include "AABB.h"
#include "CudaRenderer.h"

namespace Hitables
{

	struct PrimitiveList
	{
		Vec3*	 Centers;
		float*	 Radii;
		AABB*	 AABBs;
		uint16_t Count = 0;
	};

	__host__ inline PrimitiveList* Init(const uint32_t capacity)
	{
		PrimitiveList h_HittableList;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.Centers, capacity * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.Radii, capacity * sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.AABBs, capacity * sizeof(AABB)));

		PrimitiveList* d_HittableList;
		// Allocate memory for BVHNode structure on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&d_HittableList, sizeof(PrimitiveList)));

		// Copy initialized BVHNode data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_HittableList, &h_HittableList, sizeof(PrimitiveList), cudaMemcpyHostToDevice));
		return d_HittableList;
	}

	__device__ inline void Add(const Vec3& position, const float radius)
	{
		d_Params.List->Centers[d_Params.List->Count] = position;
		d_Params.List->Radii[d_Params.List->Count]	 = radius;
		d_Params.List->AABBs[d_Params.List->Count]	 = AABB(position - radius, position + radius);
		d_Params.List->Count++;
	}

	__device__ inline bool IntersectPrimitive(const Ray& ray, const float tMin, float& tMax, HitRecord& record, const uint16_t sphereIndex)
	{
		const Vec3	oc			 = ray.Origin - d_Params.List->Centers[sphereIndex];
		const float a			 = glm::dot(ray.Direction, ray.Direction);
		const float b			 = glm::dot(oc, ray.Direction);
		const float c			 = glm::dot(oc, oc) - d_Params.List->Radii[sphereIndex] * d_Params.List->Radii[sphereIndex];
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
		record.Normal		  = (record.Location - d_Params.List->Centers[sphereIndex]) * (1.0f / d_Params.List->Radii[sphereIndex]); // Normalize using multiplication
		record.PrimitiveIndex = sphereIndex;

		tMax = t; // Update tMax for the next intersection test
		return true;
	}

} // namespace
