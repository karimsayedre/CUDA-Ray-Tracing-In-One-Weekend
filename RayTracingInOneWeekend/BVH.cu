#include "pch.cuh"
#include <thrust/sort.h>
#include "BVH.h"

__device__ uint32_t BVHSoA::BuildBVH_SoA(const HittableList* list, uint32_t* indices, uint32_t start, uint32_t end)
{
	uint32_t object_span = end - start;

	// Compute bounding box for this node
	AABB box;
	bool first_box = true;
	for (uint32_t i = start; i < end; ++i)
	{
		uint32_t sphere_index = indices[i];
		const AABB&	 current_box  = list->m_AABB[sphere_index];
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
		return AddNode(indices[start], UINT32_MAX, box, true);
	}

	if (object_span == 2)
	{
		uint32_t idx_a = indices[start];
		uint32_t idx_b = indices[start + 1];

		// Create leaf nodes for each sphere
		const AABB& box_a = list->m_AABB[idx_a];
		const AABB& box_b = list->m_AABB[idx_b];

		uint32_t left_leaf	= AddNode(idx_a, UINT32_MAX, box_a, true);
		uint32_t right_leaf = AddNode(idx_b, UINT32_MAX, box_b, true);

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
	uint32_t	   mid		 = start + object_span / 2;
	const uint32_t left_idx	 = BuildBVH_SoA(list, indices, start, mid);
	const uint32_t right_idx = BuildBVH_SoA(list, indices, mid, end);

	AABB combined = AABB(
		m_bounds_min[left_idx],
		m_bounds_max[left_idx]);

	AABB rightAABB = AABB(
		m_bounds_min[right_idx],
		m_bounds_max[right_idx]);

	// Compute the new combined bounding box
	combined = AABB(combined, rightAABB);
	return AddNode(left_idx, right_idx, combined, false);
}

__device__ bool BVHSoA::TraverseBVH_SoA(const Ray& ray, Float tmin, Float tmax, HittableList* __restrict__ list, uint32_t root_index, HitRecord& best_hit) const
{
	bool  hit_anything	 = false;
	Float closest_so_far = tmax; // Track current best hit distance

	uint32_t stack[10]; // Reduced from 64 if your scenes don't need deep traversal
	int		 stack_ptr = 0;

	stack[stack_ptr++] = root_index;

	while (stack_ptr > 0)
	{
		uint32_t entry = stack[--stack_ptr];

		// Early exit if this node can't possibly be closer than our best hit
		if (!IntersectBounds(ray, entry, tmin, closest_so_far))
			continue;

		if (m_is_leaf[entry])
		{
			// Test intersection with the primitive
			HitRecord temp_rec;
			uint32_t  sphere_index = m_left[entry];

			// Use direct access to sphere components instead of fetching the whole object
			if (list->HitSphere(sphere_index, ray, tmin, closest_so_far, temp_rec))
			{
				closest_so_far = temp_rec.T;
				best_hit	   = temp_rec;
				hit_anything   = true;
			}
		}
		else
		{
			stack[stack_ptr++] = m_right[entry];
			stack[stack_ptr++] = m_left[entry];
		}
	}

	return hit_anything;
}