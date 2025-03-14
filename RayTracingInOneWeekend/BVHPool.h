#pragma once
#include "pch.cuh"

class BVHNode;

class BVHPool
{
  public:
	__device__ BVHPool(size_t capacity);

	~BVHPool();

	__device__ void Resize(size_t newCapacity);

	__device__ BVHNode* Allocate();
	__device__ size_t	GetUsedCount() const
	{
		return m_UsedCount;
	}

  private:
	BVHNode* m_Nodes;
	size_t	 m_Capacity;
	size_t	 m_UsedCount;
};