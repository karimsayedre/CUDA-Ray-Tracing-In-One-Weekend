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
		cuda::std::signbit(invDir.x),
		cuda::std::signbit(invDir.y),
		cuda::std::signbit(invDir.z),
	};

	stackData[stackPtr++] = currentNode;

	// Iterative traversal without immediate push of both children
	while (stackPtr != 0)
	{
		const BVH& node = m_BVHs[currentNode];
		// Process current node
		if (node.m_is_leaf)
		{
			// Leaf node - process hit test
			hit_anything |= list->Hit(ray, tmin, tmax, best_hit, node.m_left);

			// Pop next node from stack
			currentNode = stackData[--stackPtr];
			continue;
		}

		// Check both children for intersection
		bool hitLeft  = IntersectBoundsFast(ray.Origin(), invDir, dirIsNeg, node.m_left, tmin, tmax);
		bool hitRight = IntersectBoundsFast(ray.Origin(), invDir, dirIsNeg, node.m_right, tmin, tmax);

		// Neither child was hit, pop from stack
		if (!hitLeft && !hitRight)
		{
			currentNode = stackData[--stackPtr];
			continue;
		}

		if (hitLeft && hitRight)
		{
			currentNode			  = node.m_left;
			stackData[stackPtr++] = node.m_right;
			continue;
		}

		// Only one child was hit
		currentNode = hitLeft ? node.m_left : node.m_right;
	}

	return hit_anything;
}