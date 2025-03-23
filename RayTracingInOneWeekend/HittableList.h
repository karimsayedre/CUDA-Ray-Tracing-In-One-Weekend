#pragma once

#include "AABB.h"
#include "CudaRenderer.cuh"

struct HittableList
{
  public:
	__host__ inline static void Init(HittableList*& d_HittableList, const uint32_t capacity)
	{
		HittableList h_HittableList;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_Center, capacity * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_Radius, capacity * sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_AABB, capacity * sizeof(AABB)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_MaterialIndex, capacity * sizeof(uint32_t)));

		// Allocate memory for BVH structure on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&d_HittableList, sizeof(HittableList)));

		// Copy initialized BVH data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_HittableList, &h_HittableList, sizeof(HittableList), cudaMemcpyHostToDevice));
	}

	__device__ void Add(const Vec3& position, const float radius)
	{
		m_Center[m_Count]		 = position;
		m_Radius[m_Count]		 = radius;
		m_AABB[m_Count]			 = AABB(position - radius, position + radius);
		m_MaterialIndex[m_Count] = m_Count;
		m_Count++;
	}

	__device__ bool Hit(const Ray& ray, const float tMin, float& tMax, HitRecord& record, const uint32_t sphereIndex) const
	{
		const Vec3	oc			 = ray.Origin() - m_Center[sphereIndex];
		const float a			 = glm::dot(ray.Direction(), ray.Direction());
		const float b			 = glm::dot(oc, ray.Direction());
		const float c			 = glm::dot(oc, oc) - m_Radius[sphereIndex] * m_Radius[sphereIndex];
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
		record.T			 = t;
		record.Location		 = ray.PointAtT(t);
		record.Normal		 = (record.Location - m_Center[sphereIndex]) * (1.0f / m_Radius[sphereIndex]); // Normalize using multiplication
		record.MaterialIndex = m_MaterialIndex[sphereIndex];

		tMax = t;

		return true;
	}

	__device__ [[nodiscard]] uint16_t GetPrimitiveCount() const
	{
		return m_Count;
	}

  private:
	Vec3*	  m_Center;
	float*	  m_Radius;
	uint16_t* m_MaterialIndex;
	AABB*	  m_AABB;

	uint16_t m_Count = 0;

	friend struct BVHSoA;
	friend struct BVHSoARopes;
};
