#pragma once

#include "AABB.h"
#include "CudaRenderer.cuh"

namespace Hitables
{

	struct HittableList
	{
		Vec3*	 Centers;
		float*	 Radii;
		AABB*	 AABBs;
		uint16_t Count = 0;
	};

	__host__ inline void InitHittableList(HittableList*& d_HittableList, const uint32_t capacity)
	{
		HittableList h_HittableList;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.Centers, capacity * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.Radii, capacity * sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.AABBs, capacity * sizeof(AABB)));

		// Allocate memory for BVH structure on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&d_HittableList, sizeof(HittableList)));

		// Copy initialized BVH data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_HittableList, &h_HittableList, sizeof(HittableList), cudaMemcpyHostToDevice));
	}

	__device__ inline void Add(HittableList* list, const Vec3& position, const float radius)
	{
		list->Centers[list->Count] = position;
		list->Radii[list->Count]   = radius;
		list->AABBs[list->Count]   = AABB(position - radius, position + radius);
		list->Count++;
	}

	__device__ inline bool IntersectPrimitive(const HittableList* list, const Ray& ray, const float tMin, float& tMax, HitRecord& record, const uint16_t sphereIndex)
	{
		const Vec3	oc			 = ray.Origin() - list->Centers[sphereIndex];
		const float a			 = glm::dot(ray.Direction(), ray.Direction());
		const float b			 = glm::dot(oc, ray.Direction());
		const float c			 = glm::dot(oc, oc) - list->Radii[sphereIndex] * list->Radii[sphereIndex];
		const float discriminant = b * b - a * c;

		if (discriminant <= 0.0f)
			return false; // No intersection

		const float sqrtD = sqrtf(discriminant);
		const float t0	  = (-b - sqrtD) / a;
		const float t1	  = (-b + sqrtD) / a;

		// Pick the closest valid t
		const float t = (t0 > tMin && t0 < tMax) ? t0 : ((t1 > tMin && t1 < tMax) ? t1 : -1.0f);
		if (t < 0.0f)
			return false;

		// Compute intersection data
		record.Location		  = ray.PointAtT(t);
		record.Normal		  = (record.Location - list->Centers[sphereIndex]) * (1.0f / list->Radii[sphereIndex]); // Normalize using multiplication
		record.PrimitiveIndex = sphereIndex;

		tMax = t; // Update tMax for the next intersection test
		return true;
	}

} // namespace
