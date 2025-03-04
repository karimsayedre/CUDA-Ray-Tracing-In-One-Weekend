#pragma once

#include "Hittable.h"
#include "HittableList.h"

__device__ inline bool box_compare(const Hittable* a, const Hittable* b, int axis)
{
	return a->GetBoundingBox(0.0, 1.0).axis(axis).min < b->GetBoundingBox(0.0, 1.0).axis(axis).min;
}

__device__ inline bool box_x_compare(const Hittable* a, const Hittable* b)
{
	return box_compare(a, b, 0);
}

__device__ inline bool box_y_compare(const Hittable* a, const Hittable* b)
{
	return box_compare(a, b, 1);
}

__device__ inline bool box_z_compare(const Hittable* a, const Hittable* b)
{
	return box_compare(a, b, 2);
}

class BVHNode : public Hittable
{
  public:
	__device__ BVHNode(const HittableList& list, double time0, double time1, curandState* local_rand_state)
		: BVHNode(list.m_Objects, 0, list.m_Count, time0, time1, local_rand_state)
	{
	}

	__device__ BVHNode(Hittable** src_objects, size_t start, size_t end, double time0, double time1, curandState* local_rand_state)
	{
		size_t object_span = end - start;

		// Allocate temporary device array
		Hittable** objects = new Hittable*[object_span];

		// Copy objects into the temporary array
		for (size_t i = 0; i < object_span; i++)
		{
			objects[i] = src_objects[start + i];
		}

		// Build the bounding box of the span of source objects.
		m_Box = {};
		for (size_t object_index = 0; object_index < object_span; object_index++)
			m_Box = AABB(m_Box, objects[object_index]->GetBoundingBox(0.0, 1.0));

		int	 axis		= m_Box.LongestAxis();
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

			m_Left	= new BVHNode(objects, 0, mid, time0, time1, local_rand_state);
			m_Right = new BVHNode(objects, mid, object_span, time0, time1, local_rand_state);
		}

		// Compute Bounding Box
		AABB box_left  = m_Left->GetBoundingBox(time0, time1);
		AABB box_right = m_Right->GetBoundingBox(time0, time1);

		m_Box = AABB(box_left, box_right);

		// Free temporary array
		delete[] objects;
	}

__device__ __noinline__ bool Hit(const Ray& r, const Float tMin, Float tMax, HitRecord& rec) const override
	{
		Hittable* __restrict__ stack[10]; // Fixed-size traversal stack
		unsigned char stack_ptr		 = 0; // Use an 8-bit integer since depth <= 10
		bool		  hit_anything	 = false;
		Float		  closest_so_far = tMax; // Local variable for the closest hit

		// Push root children (right first, then left so left is processed first)
		stack[stack_ptr++] = m_Right;
		stack[stack_ptr++] = m_Left;

		HitRecord temp_rec; // Declare once outside the loop to reuse the register(s)

		while (stack_ptr > 0)
		{
			Hittable* node = stack[--stack_ptr];

			// Early out: Skip nodes whose AABB doesn't intersect [tMin, closest_so_far]
			const AABB& box = node->GetBoundingBox(0.0, 1.0);
			if (!box.Hit(r, {tMin, closest_so_far}))
				continue;

			if (node->IsLeaf())
			{
				if (node->Hit(r, tMin, closest_so_far, temp_rec))
				{
					hit_anything   = true;
					closest_so_far = temp_rec.T;
					rec			   = temp_rec;
				}
			}
			else
			{
				BVHNode* bvh_node = static_cast<BVHNode*>(node);
				// Push children without sorting; order remains consistent.
				stack[stack_ptr++] = bvh_node->m_Right;
				stack[stack_ptr++] = bvh_node->m_Left;
			}
		}
		return hit_anything;
	}


	__device__ const AABB& GetBoundingBox(double time0, double time1) const override
	{
		return m_Box;
	}

	__device__ [[nodiscard]] bool IsLeaf() const override
	{
		return false;
	}

  private:
	Hittable* m_Left;
	Hittable* m_Right;
	AABB	  m_Box;
};
