#pragma once

#include "AABB.h"
#include "CudaRenderer.cuh"


struct HittableList
{
  public:
	__host__ inline static void Init(HittableList*& d_hittableList, uint32_t maxObjs)
	{
		HittableList h_HittableList;
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_Center, maxObjs * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_Radius, maxObjs * sizeof(Float)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_AABB, maxObjs * sizeof(AABB)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_HittableList.m_MaterialIndex, maxObjs * sizeof(uint32_t)));

		// Allocate memory for BVH structure on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&d_hittableList, sizeof(HittableList)));

		// Copy initialized BVH data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_hittableList, &h_HittableList, sizeof(HittableList), cudaMemcpyHostToDevice));
	}

	__device__ bool Hit(const Ray& ray, const Float tMin, Float tMax, HitRecord& record, uint32_t sphereIndex) const
	{
		Vec3  oc		   = ray.Origin() - m_Center[sphereIndex];
		Float a			   = dot(ray.Direction(), ray.Direction());
		Float b			   = dot(oc, ray.Direction());
		Float c			   = dot(oc, oc) - m_Radius[sphereIndex] * m_Radius[sphereIndex];
		Float discriminant = b * b - a * c;

		if (discriminant <= 0.0f)
			return false; // No intersection 

		Float sqrtD = sqrtf(discriminant);
		Float t0	= (-b - sqrtD) / a;
		Float t1	= (-b + sqrtD) / a;

		// Pick the closest valid t
		Float t = (t0 > tMin && t0 < tMax) ? t0 : ((t1 > tMin && t1 < tMax) ? t1 : -1.0f);
		if (t < 0.0f)
			return false;

		// Compute intersection data
		record.T			 = t;
		record.Location		 = ray.point_at_parameter(t);
		record.Normal		 = (record.Location - m_Center[sphereIndex]) * (1.0f / m_Radius[sphereIndex]); // Normalize using multiplication
		record.MaterialIndex = m_MaterialIndex[sphereIndex];

		return true;
	}

	__device__ void Add(const Vec3& position, float radius)
	{
		m_Center[m_Count]		 = position;
		m_Radius[m_Count]		 = radius;
		m_AABB[m_Count]			 = AABB(position - radius, position + radius);
		m_MaterialIndex[m_Count] = m_Count;
		m_Count++;
	}

	Vec3*	  m_Center;
	Float*	  m_Radius;
	uint16_t* m_MaterialIndex;
	AABB*	  m_AABB;

	uint16_t m_Count = 0;
};
