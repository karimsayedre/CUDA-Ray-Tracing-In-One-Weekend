#pragma once

#include "BVH.h"
#include "pch.cuh"

#if 0
__device__ inline void DebugBVHNode(BVHSoA* soa, uint32_t nodeIndex, int depth = 0)
{
	if (nodeIndex >= soa->m_Count)
	{
		printf("Invalid node index: %u\n", nodeIndex);
		return;
	}

	// Print indentation
	for (int i = 0; i < depth; i++)
		printf("  ");

	if (soa->m_is_leaf[nodeIndex])
	{
		printf("Leaf Node %u: Sphere %u, Bounds: (%.2f,%.2f,%.2f) to (%.2f,%.2f,%.2f)\n",
			   nodeIndex,
			   soa->m_left[nodeIndex],
			   soa->m_bounds_min[nodeIndex].x,
			   soa->m_bounds_min[nodeIndex].y,
			   soa->m_bounds_min[nodeIndex].z,
			   soa->m_bounds_max[nodeIndex].x,
			   soa->m_bounds_max[nodeIndex].y,
			   soa->m_bounds_max[nodeIndex].z);
	}
	else
	{
		printf("Internal Node %u: Left %u, Right %u, Bounds: (%.2f,%.2f,%.2f) to (%.2f,%.2f,%.2f)\n",
			   nodeIndex,
			   soa->m_left[nodeIndex],
			   soa->Right[nodeIndex],
			   soa->m_bounds_min[nodeIndex].x,
			   soa->m_bounds_min[nodeIndex].y,
			   soa->m_bounds_min[nodeIndex].z,
			   soa->m_bounds_max[nodeIndex].x,
			   soa->m_bounds_max[nodeIndex].y,
			   soa->m_bounds_max[nodeIndex].z);

		// Recursively print children
		if (depth < 10)
		{ // Limit depth to avoid infinite recursion
			DebugBVHNode(soa, soa->m_left[nodeIndex], depth + 1);
			DebugBVHNode(soa, soa->Right[nodeIndex], depth + 1);
		}
	}
}

// Add this to your ray tracing kernel to debug traversal
__device__ inline void DebugTraversal(const Ray& ray, BVHSoA* world, HittableList* list, uint32_t pixel_index)
{
	if (pixel_index == 0)
	{ // Only debug for the first pixel
		printf("Ray origin: (%.2f,%.2f,%.2f), direction: (%.2f,%.2f,%.2f)\n",
			   ray.Origin().x,
			   ray.Origin().y,
			   ray.Origin().z,
			   ray.Direction().x,
			   ray.Direction().y,
			   ray.Direction().z);

		HitRecord rec;
		bool	  hit = world->Traverse(ray, 0.001f, FLT_MAX, list, world->root, rec);
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
				   rec.MaterialIndex);
		}
	}
}
#endif
