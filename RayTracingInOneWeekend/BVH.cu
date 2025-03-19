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
		AABB	 current_box  = list->m_Objects[sphere_index].GetBoundingBox();
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
		AABB box_a = list->m_Objects[idx_a].GetBoundingBox();
		AABB box_b = list->m_Objects[idx_b].GetBoundingBox();

		uint32_t left_leaf	= AddNode(idx_a, UINT32_MAX, box_a, true);
		uint32_t right_leaf = AddNode(idx_b, UINT32_MAX, box_b, true);

		// Create parent internal node
		AABB combined = AABB(box_a, box_b);
		return AddNode(left_leaf, right_leaf, combined, false);
	}

	// Sort indices based on bounding box centroids
	int	 axis		= box.LongestAxis();
	auto comparator = [&](uint32_t a, uint32_t b)
	{
		return list->m_Objects[a].GetBoundingBox().Center()[axis] < list->m_Objects[b].GetBoundingBox().Center()[axis];
	};

	// Sort the indices - THIS IS IMPORTANT!
	// You need to implement this sorting, either using thrust::sort or another method
	// For example, a simple bubble sort (not efficient, but for illustration):
	for (uint32_t i = start; i < end; i++)
	{
		for (uint32_t j = start; j < end - 1; j++)
		{
			if (comparator(indices[j + 1], indices[j]))
			{
				uint32_t temp  = indices[j];
				indices[j]	   = indices[j + 1];
				indices[j + 1] = temp;
			}
		}
	}

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

__device__ bool BVHSoA::TraverseBVH_SoA(const Ray& ray, float tmin, float tmax, HittableList* __restrict__ list, uint32_t root_index, HitRecord& best_hit)
{
	bool  hit_anything	 = false;
	float closest_so_far = tmax; // Track current best hit distance

	uint32_t stack[10]; // Reduced from 64 if your scenes don't need deep traversal
	int		   stack_ptr = 0;

	stack[stack_ptr++] = {root_index};

	// Near the beginning of the while loop
	//if (stack_ptr < 30)
	//{ // Make sure we have room in the stack
	//	// Prefetch the next nodes that we might need
	//	uint32_t prefetch_idx = stack[stack_ptr - 1].node_idx;
	//	__prefetch(&soa->m_is_leaf[prefetch_idx], __CUDA_PREFETCH_L1);
	//	__prefetch(&soa->m_left[prefetch_idx], __CUDA_PREFETCH_L1);
	//	__prefetch(&soa->m_right[prefetch_idx], __CUDA_PREFETCH_L1);
	//}

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
			// Traverse children in front-to-back order based on ray direction
			//uint32_t left  = m_left[entry];
			//uint32_t right = m_right[entry];

			// Simple heuristic - determine traversal order based on ray direction
			// and split axis (assuming split axis is stored or derivable)
			//uint32_t split_axis			 = GetSplitAxis(entry.node_idx);
			//bool	 traverse_left_first = ray.Direction()[split_axis] < 0.0f;

			//if (traverse_left_first)
			//{
			stack[stack_ptr++] = m_right[entry];
			stack[stack_ptr++] = m_left[entry];
			//}
			//else
			{
				//stack[stack_ptr++] = {left};
				//stack[stack_ptr++] = {right};
			}
		}
	}

	return hit_anything;
}