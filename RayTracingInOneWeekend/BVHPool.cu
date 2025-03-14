#include "pch.cuh"

#include "BVHPool.h"
#include "BVH.h"


__device__ BVHPool::BVHPool(size_t capacity)
{
	m_Capacity  = capacity;
	m_UsedCount = 0;
	m_Nodes     = (BVHNode*)malloc(capacity * sizeof(BVHNode));
}

BVHPool::~BVHPool()
{
	delete[] m_Nodes;
}

__device__ void BVHPool::Resize(size_t newCapacity)
{
	BVHNode* newNodes = (BVHNode*)malloc(newCapacity * sizeof(BVHNode));
	memcpy(newNodes, m_Nodes, m_UsedCount * sizeof(BVHNode));
	delete[] m_Nodes;
	m_Nodes    = newNodes;
	m_Capacity = newCapacity;
}

__device__ BVHNode* BVHPool::Allocate()
{
	if (m_UsedCount >= m_Capacity)
	{
		Resize(m_UsedCount);
		return &m_Nodes[m_UsedCount++];
	}
	return &m_Nodes[m_UsedCount++];
}