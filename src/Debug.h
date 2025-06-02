#pragma once

#include "pch.h"
#include "BVH.h"
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include "Ray.h"
#include "Renderer.h"

#ifdef RTIOW_DEBUG_BVH
__device__ __host__ CPU_ONLY_INLINE void DebugBVHNode(BVH::BVH* soa, const uint32_t nodeIndex, const int depth = 0)
{
	if (nodeIndex >= soa->m_Count)
	{
		printf("Invalid node index: %u\n", nodeIndex);
		return;
	}

	// Print indentation
	for (int i = 0; i < depth; i++)
		printf("  ");

	if (soa->m_Nodes[nodeIndex].Right == UINT32_MAX)
	{
		printf("Leaf Node %u: Sphere %u, Bounds: (%.2f,%.2f,%.2f) to (%.2f,%.2f,%.2f)\n",
			   nodeIndex,
			   soa->m_Nodes[nodeIndex].Left,
			   soa->m_Bounds[nodeIndex].Min.x,
			   soa->m_Bounds[nodeIndex].Min.y,
			   soa->m_Bounds[nodeIndex].Min.z,
			   soa->m_Bounds[nodeIndex].Max.x,
			   soa->m_Bounds[nodeIndex].Max.y,
			   soa->m_Bounds[nodeIndex].Max.z);
	}
	else
	{
		printf("Internal Node %u: Left %u, Right %u, Bounds: (%.2f,%.2f,%.2f) to (%.2f,%.2f,%.2f)\n",
			   nodeIndex,
			   soa->m_Nodes[nodeIndex].Left,
			   soa->m_Nodes[nodeIndex].Right,
			   soa->m_Bounds[nodeIndex].Min.x,
			   soa->m_Bounds[nodeIndex].Min.y,
			   soa->m_Bounds[nodeIndex].Min.z,
			   soa->m_Bounds[nodeIndex].Max.x,
			   soa->m_Bounds[nodeIndex].Max.y,
			   soa->m_Bounds[nodeIndex].Max.z);

		// Recursively print children
		if (depth < 10)
		{ // Limit depth to avoid infinite recursion
			DebugBVHNode(soa, soa->m_Nodes[nodeIndex].Left, depth + 1);
			DebugBVHNode(soa, soa->m_Nodes[nodeIndex].Right, depth + 1);
		}
	}
}

// Add this to your ray tracing kernel to debug traversal
__device__ __host__ CPU_ONLY_INLINE void DebugTraversal(const Ray& ray, const uint32_t pixelIndex)
{
	if (pixelIndex == 0)
	{ // Only debug for the first pixel
		printf("Ray origin: (%.2f,%.2f,%.2f), direction: (%.2f,%.2f,%.2f)\n",
			   ray.Origin.x,
			   ray.Origin.y,
			   ray.Origin.z,
			   ray.Direction.x,
			   ray.Direction.y,
			   ray.Direction.z);

		HitRecord  rec;
		const bool hit = BVH::Traverse(ray, 0.001f, FLT_MAX, rec);
		printf("Hit result: %s\n", hit ? "true" : "false");

		if (hit)
		{
			printf("Hit point: (%.2f,%.2f,%.2f), normal: (%.2f,%.2f,%.2f), material: %u\n",
				   rec.Location.x,
				   rec.Location.y,
				   rec.Location.z,
				   rec.Normal.x,
				   rec.Normal.y,
				   rec.Normal.z,
				   rec.PrimitiveIndex);
		}
	}
}
#endif
