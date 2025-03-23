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
	bool hit_anything = false;

	uint16_t stack[6]; // Reduced from 64 if your scenes don't need deep traversal
	int		 stack_ptr = 0;

	stack[stack_ptr++]	   = root;
	const Vec3 invDir	   = 1.0f / ray.Direction();
	const int  dirIsNeg[3] = {
		 std::signbit(invDir.x),
		 std::signbit(invDir.y),
		 std::signbit(invDir.z),
	 };
	while (stack_ptr > 0)
	{
		uint16_t entry = stack[--stack_ptr];

		// Early exit if this node can't possibly be closer than our best hit
		if (!IntersectBoundsFast(ray.Origin(), invDir, dirIsNeg, entry, tmin, tmax))
			continue;

		if (m_is_leaf[entry])
		{
			hit_anything |= list->Hit(ray, tmin, tmax, best_hit, m_left[entry]);
		}
		else
		{
			stack[stack_ptr++] = m_right[entry];
			stack[stack_ptr++] = m_left[entry];
		}
	}

	return hit_anything;
}