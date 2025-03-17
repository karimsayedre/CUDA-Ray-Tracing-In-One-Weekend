#include "pch.cuh"
#include <thrust/sort.h>
#include "BVH.h"

__device__ uint32_t BuildBVH_SoA(const HittableList* list, uint32_t* indices, uint32_t start, uint32_t end, BVHSoA* soa)
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
		return soa->AddNode(indices[start], UINT32_MAX, box, true);
	}

	if (object_span == 2)
	{
		uint32_t idx_a = indices[start];
		uint32_t idx_b = indices[start + 1];

		// Create leaf nodes for each sphere
		AABB box_a = list->m_Objects[idx_a].GetBoundingBox();
		AABB box_b = list->m_Objects[idx_b].GetBoundingBox();

		uint32_t left_leaf	= soa->AddNode(idx_a, UINT32_MAX, box_a, true);
		uint32_t right_leaf = soa->AddNode(idx_b, UINT32_MAX, box_b, true);

		// Create parent internal node
		AABB combined = AABB(box_a, box_b);
		return soa->AddNode(left_leaf, right_leaf, combined, false);
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
	const uint32_t left_idx	 = BuildBVH_SoA(list, indices, start, mid, soa);
	const uint32_t right_idx = BuildBVH_SoA(list, indices, mid, end, soa);

	// Create internal node
	AABB combined = AABB(soa->m_bounds[left_idx], soa->m_bounds[right_idx]);
	return soa->AddNode(left_idx, right_idx, combined, false);
}

__device__ bool TraverseBVH_SoA(const Ray& ray, float tmin, float tmax, HittableList* list, BVHSoA* soa, uint32_t root_index, HitRecord& best_hit)
{
	bool hit_anything = false;

	// Push root node
	struct StackEntry
	{
		uint32_t index;
		float	 Tmin, Tmax;
	};

	StackEntry stack[10]; // Increased stack size for deeper trees
	int		   stack_ptr = 0;
	stack[stack_ptr++]	 = {root_index, tmin, tmax};

	while (stack_ptr > 0)
	{
		StackEntry entry = stack[--stack_ptr];

		// Test if the ray intersects this node's bounding box
		const AABB& box = soa->m_bounds[entry.index];
		if (!box.Hit(ray, {entry.Tmin, entry.Tmax}))
			continue;

		if (soa->m_is_leaf[entry.index])
		{
			// It's a leaf node, test intersection with the sphere
			HitRecord	  temp_rec;
			uint32_t	  sphere_index = soa->m_left[entry.index]; // Sphere index stored in left child
			const Sphere& sphere	   = list->m_Objects[sphere_index];

			if (sphere.Hit(ray, entry.Tmin, entry.Tmax, temp_rec))
			{
				// We found a hit, update tmax and record
				hit_anything = true;
				entry.Tmax	 = temp_rec.T; // Update tmax for next intersections
				best_hit	 = temp_rec;
				tmax		 = temp_rec.T; // Update global tmax
			}
		}
		else
		{
			// It's an internal node, push both children
			// Push right child first, then left (so left is processed first)
			uint32_t left_child	 = soa->m_left[entry.index];
			uint32_t right_child = soa->m_right[entry.index];

			// Push right first (will be processed second)
			stack[stack_ptr++] = {right_child, entry.Tmin, entry.Tmax};
			// Push left (will be processed first)
			stack[stack_ptr++] = {left_child, entry.Tmin, entry.Tmax};
		}
	}

	return hit_anything;
}
