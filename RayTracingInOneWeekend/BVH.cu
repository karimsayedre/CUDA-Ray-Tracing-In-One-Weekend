#include "pch.cuh"
#include "BVH.h"

#include "AABB.h"
#include "Ray.h"

__device__ uint16_t BVHSoA::BuildBVH_SoA(const HittableList* list, uint16_t* indices, uint16_t start, uint16_t end)
{
	uint16_t object_span = end - start;

	// Compute bounding box for this node
	AABB box;
	bool first_box = true;
	for (uint16_t i = start; i < end; ++i)
	{
		uint32_t	sphere_index = indices[i];
		const AABB& current_box	 = list->m_AABB[sphere_index];
		if (first_box)
		{
			box		  = current_box;
			first_box = false;
		}
		else
		{
			box = AABB(box, current_box);
		}
	}

	if (object_span == 1)
	{
		// Leaf node: store sphere index
		return AddNode(indices[start], UINT16_MAX, box, true);
	}

	if (object_span == 2)
	{
		uint16_t idx_a = indices[start];
		uint16_t idx_b = indices[start + 1];

		// Create leaf nodes for each sphere
		const AABB& box_a = list->m_AABB[idx_a];
		const AABB& box_b = list->m_AABB[idx_b];

		uint16_t left_leaf	= AddNode(idx_a, UINT16_MAX, box_a, true);
		uint16_t right_leaf = AddNode(idx_b, UINT16_MAX, box_b, true);

		// Create parent internal node
		const AABB combined(box_a, box_b);
		return AddNode(left_leaf, right_leaf, combined, false);
	}

	// Sort indices based on bounding box centroids
	int	 axis		= box.LongestAxis();
	auto comparator = [&](uint32_t a, uint32_t b)
	{
		return list->m_AABB[a].Center()[axis] < list->m_AABB[b].Center()[axis];
	};

	thrust::sort(indices + start, indices + end, comparator);

	// Recursively build children
	uint16_t	   mid		 = start + object_span / 2;
	const uint16_t left_idx	 = BuildBVH_SoA(list, indices, start, mid);
	const uint16_t right_idx = BuildBVH_SoA(list, indices, mid, end);

	AABB combined = m_bounds[left_idx];

	const AABB& rightAABB = m_bounds[right_idx];

	// Compute the new combined bounding box
	combined = AABB(combined, rightAABB);
	return AddNode(left_idx, right_idx, combined, false);
}

__device__ bool BVHSoA::TraverseBVH_SoA(const Ray& ray, Float tmin, Float tmax, HittableList* __restrict__ list, HitRecord& best_hit) const
{
	// Use registers for stack instead of memory
	uint16_t currentNode = root;
	uint16_t stackData[6]; // Reduced stack size - 6 is often sufficient for most scenes
	int		 stackPtr	  = 0;
	bool	 hit_anything = false;

	// Pre-compute ray inverse direction and signs for faster bounds tests
	const Vec3 invDir = {
		__float2half(1.0f) / ray.Direction().x,
		__float2half(1.0f) / ray.Direction().y,
		__float2half(1.0f) / ray.Direction().z};
	const int dirIsNeg[3] = {
		invDir.x < __float2half(0.0f) ? 1 : 0,
		invDir.y < __float2half(0.0f) ? 1 : 0,
		invDir.z < __float2half(0.0f) ? 1 : 0};

	// Iterative traversal without immediate push of both children
	while (true)
	{
		// Process current node
		if (m_is_leaf[currentNode])
		{
			// Leaf node - process hit test
			hit_anything |= list->Hit(ray, tmin, tmax, best_hit, m_left[currentNode]);

			// Pop next node from stack
			if (stackPtr == 0)
				break;
			currentNode = stackData[--stackPtr];
			continue;
		}

		// Interior node - fetch children
		uint16_t leftChild	= m_left[currentNode];
		uint16_t rightChild = m_right[currentNode];

		// Check both children for intersection
		bool hitLeft  = IntersectBoundsFast(ray.Origin(), invDir, dirIsNeg, leftChild, tmin, tmax);
		bool hitRight = IntersectBoundsFast(ray.Origin(), invDir, dirIsNeg, rightChild, tmin, tmax);

		// Neither child was hit, pop from stack
		if (!hitLeft && !hitRight)
		{
			if (stackPtr == 0)
				break;
			currentNode = stackData[--stackPtr];
			continue;
		}

		// Both children were hit, process closer one first
		if (hitLeft && hitRight)
		{
			// Determine traversal order (closer one first)
			// This heuristic improves ray termination
			float leftDist	= ComputeNodeDistance(ray, leftChild);
			float rightDist = ComputeNodeDistance(ray, rightChild);

			if (leftDist < rightDist)
			{
				// Left is farther, process right first
				currentNode			  = rightChild;
				stackData[stackPtr++] = leftChild; // Fixed: removed stack size check
			}
			else
			{
				// Right is farther, process left first
				currentNode			  = leftChild;
				stackData[stackPtr++] = rightChild; // Fixed: removed stack size check
			}
			continue;
		}

		// Only one child was hit
		currentNode = hitLeft ? leftChild : rightChild;
	}

	return hit_anything;
}