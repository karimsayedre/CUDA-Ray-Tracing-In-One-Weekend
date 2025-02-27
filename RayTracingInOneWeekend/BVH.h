#pragma once

#include "Hittable.h"
#include "HittableList.h"

class BVHNode : public Hittable
{
  public:
	__device__ BVHNode(const HittableList& list, double time0, double time1, curandState* local_rand_state);

	__device__ BVHNode(Hittable** src_objects, size_t start, size_t end, double time0, double time1, curandState* local_rand_state);

	__device__ bool Hit(const Ray& r, const Float tMin, Float tMax, HitRecord& rec) const
	{
		Hittable* stack[50]; // Adjust stack depth as needed
		int		  stack_ptr		 = 0;
		bool	  hit_anything	 = false;
		Float	  closest_so_far = tMax;

		// Push root children (right first, then left to process left first)
		stack[stack_ptr++] = m_Right;
		stack[stack_ptr++] = m_Left;

		while (stack_ptr > 0)
		{
			Hittable* node = stack[--stack_ptr];

			// Early out: Skip nodes whose AABB doesn't intersect [tMin, closest_so_far]
			AABB box;
			node->GetBoundingBox(0, 0, box);
			if (!box.Hit(r, tMin, closest_so_far))
				continue;

			if (node->IsLeaf())
			{ // Assume `IsLeaf()` checks for primitive (e.g., Sphere)
				HitRecord temp_rec;
				if (node->Hit(r, tMin, closest_so_far, temp_rec))
				{
					hit_anything   = true;
					closest_so_far = temp_rec.T;
					rec			   = temp_rec;
				}
			}
			else
			{ // Internal BVHNode
				BVHNode* bvh_node = static_cast<BVHNode*>(node);
				// Push children in reverse order (right first, left next)
				stack[stack_ptr++] = bvh_node->m_Right;
				stack[stack_ptr++] = bvh_node->m_Left;
			}
		}
		return hit_anything;
	}

	__device__ bool GetBoundingBox(double time0, double time1, AABB& outputBox) const
	{
		outputBox = m_Box;
		return true;
	}

  private:
	Hittable* m_Left;
	Hittable* m_Right;
	AABB	  m_Box;
};
