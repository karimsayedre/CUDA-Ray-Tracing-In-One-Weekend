#include "pch.cuh"

#include "BVH.h"

#include <algorithm>

#include "Sphere.h"

#include "Random.h"

// void* __cdecl operator new[](size_t size, char const*, int, unsigned int, char const*, int)
//{
//	return malloc(size);
// }
//
// void * __cdecl operator new[](unsigned __int64 size,unsigned __int64,unsigned __int64,char const *,int,unsigned int,char const *,int)
//{
//	return malloc(size);
// }

__device__ inline bool box_compare(const Hittable* a, const Hittable* b, int axis)
{
	AABB box_a;
	AABB box_b;

	if (!a->GetBoundingBox(0, 0, box_a) || !b->GetBoundingBox(0, 0, box_b))
		printf("No bounding box in bvh_node constructor.");

	return box_a.Min()[axis] < box_b.Min()[axis];
}

__device__ bool box_x_compare(const Hittable* a, const Hittable* b)
{
	return box_compare(a, b, 0);
}

__device__ bool box_y_compare(const Hittable* a, const Hittable* b)
{
	return box_compare(a, b, 1);
}

__device__ bool box_z_compare(const Hittable* a, const Hittable* b)
{
	return box_compare(a, b, 2);
}

__device__ BVHNode::BVHNode(const HittableList& list, double time0, double time1, curandState* local_rand_state)
	: BVHNode(list.m_Objects, 0, list.m_Count, time0, time1, local_rand_state)
{
}

__device__ BVHNode::BVHNode(Hittable** src_objects, size_t start, size_t end, double time0, double time1, curandState* local_rand_state)
	: Hittable(this)
{
	size_t object_span = end - start;

	// Allocate temporary device array
	Hittable** objects = new Hittable*[object_span];

	// Copy objects into the temporary array
	for (size_t i = 0; i < object_span; i++)
	{
		objects[i] = src_objects[start + i];
	}

	// Pick a random axis
	int	 axis		= RandomInt(local_rand_state, 0, 2);
	auto comparator = (axis == 0)	? box_x_compare
					  : (axis == 1) ? box_y_compare
									: box_z_compare;

	if (object_span == 1)
	{
		m_Left = m_Right = objects[0];
	}
	else if (object_span == 2)
	{
		if (comparator(objects[0], objects[1]))
		{
			m_Left	= objects[0];
			m_Right = objects[1];
		}
		else
		{
			m_Left	= objects[1];
			m_Right = objects[0];
		}
	}
	else
	{
		// Replace the bubble sort with median selection:
		size_t mid = object_span / 2;
		// Partition objects around the approximate median (no full sort)
		for (size_t i = 0; i < object_span; ++i)
		{
			for (size_t j = i + 1; j < object_span; ++j)
			{
				if (comparator(objects[j], objects[i]))
				{
					Hittable* temp = objects[i];
					objects[i]	   = objects[j];
					objects[j]	   = temp;
				}
			}
			if (i == mid)
				break; // Early exit after partitioning around mid
		}

		m_Left	   = new BVHNode(objects, 0, mid, time0, time1, local_rand_state);
		m_Right	   = new BVHNode(objects, mid, object_span, time0, time1, local_rand_state);
	}

	// Compute Bounding Box
	AABB box_left, box_right;
	if (!m_Left->GetBoundingBox(time0, time1, box_left) || !m_Right->GetBoundingBox(time0, time1, box_right))
	{
		printf("No bounding box in BVHNode constructor. \n");
		// LOG_CORE_ERROR("No bounding box in BVHNode constructor.");
	}

	m_Box = SurroundingBox(box_left, box_right);

	// Free temporary array
	delete[] objects;
}
