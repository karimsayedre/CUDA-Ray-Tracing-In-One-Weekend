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

	// Handle leaf cases
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

	// Use SAH to find the best split
	float	 best_cost		= FLT_MAX;
	uint16_t best_axis		= 0;
	uint16_t best_split_idx = start + object_span / 2; // Default middle split as fallback

	// Cost of not splitting (creating a leaf)
	float no_split_cost = object_span * box.SurfaceArea();

	// Try each axis
	for (uint16_t axis = 0; axis < 3; ++axis)
	{
		// Sort indices along this axis
		auto comparator = [&](uint32_t a, uint32_t b)
		{
			return list->m_AABB[a].Center()[axis] < list->m_AABB[b].Center()[axis];
		};

		thrust::sort(indices + start, indices + end, comparator);

		// Precompute all bounding boxes from left to right
		AABB* left_boxes = new AABB[object_span];
		left_boxes[0]	 = list->m_AABB[indices[start]];
		for (uint16_t i = 1; i < object_span; ++i)
		{
			left_boxes[i] = AABB(left_boxes[i - 1], list->m_AABB[indices[start + i]]);
		}

		// Precompute all bounding boxes from right to left
		AABB* right_boxes			 = new AABB[object_span];
		right_boxes[object_span - 1] = list->m_AABB[indices[end - 1]];
		for (int i = object_span - 2; i >= 0; --i)
		{
			right_boxes[i] = AABB(right_boxes[i + 1], list->m_AABB[indices[start + i]]);
		}

		// Evaluate SAH cost for each possible split
		for (uint16_t i = 1; i < object_span; ++i)
		{
			uint16_t left_count	 = i;
			uint16_t right_count = object_span - i;

			float left_sa  = left_boxes[i - 1].SurfaceArea();
			float right_sa = right_boxes[i].SurfaceArea();

			// SAH cost formula: C = T_traverse + (left_count * left_sa + right_count * right_sa) / parent_sa * T_intersect
			// We can simplify by using constant traversal and intersection costs
			float traversal_cost	= 1.0f;
			float intersection_cost = 1.0f;
			float parent_sa			= box.SurfaceArea();

			float cost = traversal_cost + (left_count * left_sa + right_count * right_sa) / parent_sa * intersection_cost;

			if (cost < best_cost)
			{
				best_cost	   = cost;
				best_axis	   = axis;
				best_split_idx = start + i;
			}
		}

		delete[] left_boxes;
		delete[] right_boxes;
	}

	// If no split is better than not splitting, and we have fewer than some threshold of objects,
	// we could make this a leaf. However, we'll always split for simplicity and compatibility
	// with the traversal function.

	// Resort along the best axis if it's not the last one we tried
	if (best_axis != 2)
	{
		auto comparator = [&](uint32_t a, uint32_t b)
		{
			return list->m_AABB[a].Center()[best_axis] < list->m_AABB[b].Center()[best_axis];
		};

		thrust::sort(indices + start, indices + end, comparator);
	}

	// Recursively build children
	const uint16_t left_idx	 = BuildBVH_SoA(list, indices, start, best_split_idx);
	const uint16_t right_idx = BuildBVH_SoA(list, indices, best_split_idx, end);

	// Compute the combined bounding box from the actual child nodes
	AABB		combined  = m_bounds[left_idx];
	const AABB& rightAABB = m_bounds[right_idx];
	combined			  = AABB(combined, rightAABB);

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