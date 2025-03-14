#include "pch.cuh"

#include "BVH.h"

__device__ __noinline__ BVHNode::BVHNode(
	Hittable**	 src_objects,
	size_t		 start,
	size_t		 end,
	double		 time0,
	double		 time1,
	curandState* local_rand_state,
	BVHPool*	 pool)
	: Hittable(this)
{
	BVHNode* node = pool->Allocate();
	if (!node)
	{
		printf("ERROR: BVH pool exhausted!\n");
		return; // Or handle this error appropriately
	}

	size_t object_span = end - start;

	// Initialize the node's bounding box
	AABB box;
	for (size_t object_index = start; object_index < end; object_index++)
	{
		box = AABB(box, src_objects[object_index]->GetBoundingBox(time0, time1));
	}
	m_BoundingBox		  = box;
	int		   axis		  = box.LongestAxis();
	const auto comparator = (axis == 0)	  ? box_x_compare
							: (axis == 1) ? box_y_compare
										  : box_z_compare;

	if (object_span == 1)
	{
		m_Left = m_Right = src_objects[start];
	}
	else if (object_span == 2)
	{
		if (comparator(src_objects[start], src_objects[start + 1]))
		{
			m_Left	= src_objects[start];
			m_Right = src_objects[start + 1];
		}
		else
		{
			m_Left	= src_objects[start + 1];
			m_Right = src_objects[start];
		}
	}
	else
	{
		// Create a temporary array for sorting
		Hittable** objects = new Hittable*[object_span];
		for (size_t i = 0; i < object_span; i++)
		{
			objects[i] = src_objects[start + i];
		}

		// Implement efficient partitioning (modified from your original code)
		size_t mid = object_span / 2;
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

		// Recursively build left and right subtrees using the pool
		m_Left	= new BVHNode(objects, 0, mid, time0, time1, local_rand_state, pool);
		m_Right = new BVHNode(objects, mid, object_span, time0, time1, local_rand_state, pool);

		// Free temporary array
		delete[] objects;
	}

	// Node is initialized
}

__device__ __noinline__ void BVHNode::Initialize(Hittable** src_objects, size_t start, size_t end, double time0, double time1, curandState* local_rand_state, BVHPool* pool)
{
	size_t object_span = end - start;

	// Allocate temporary device array
	Hittable** objects = new Hittable*[object_span];

	// Copy objects into the temporary array
	for (size_t i = 0; i < object_span; i++)
	{
		objects[i] = src_objects[start + i];
	}

	// Build the bounding box as before
	m_BoundingBox = {};
	for (size_t object_index = 0; object_index < object_span; object_index++)
		m_BoundingBox = AABB(m_BoundingBox, objects[object_index]->GetBoundingBox(0.0, 1.0));

	int	 axis		= m_BoundingBox.LongestAxis();
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
		// Use the same median algorithm
		size_t mid = object_span / 2;
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
				break;
		}

		// Allocate from pool instead of using new
		BVHNode* left_node	= pool->Allocate();
		BVHNode* right_node = pool->Allocate();

		if (left_node && right_node)
		{
			left_node->Initialize(objects, 0, mid, time0, time1, local_rand_state, pool);
			right_node->Initialize(objects, mid, object_span, time0, time1, local_rand_state, pool);
			m_Left	= left_node;
			m_Right = right_node;
		}
		else
		{
			// Handle pool exhaustion
			// Could fall back to direct allocation or error
		}
	}

	// Compute Bounding Box
	AABB box_left  = m_Left->GetBoundingBox(time0, time1);
	AABB box_right = m_Right->GetBoundingBox(time0, time1);
	m_BoundingBox  = AABB(box_left, box_right);

	// Free temporary array
	delete[] objects;
}

__device__ bool BVHNode::processLeafNode(Hittable* node, const Ray& r, const Float tMin, Float& tMax, HitRecord& rec)
{
	HitRecord temp_rec;
	if ((Sphere*)(node)->Hit(r, tMin, tMax, temp_rec))
	{
		tMax = temp_rec.T;
		rec	 = temp_rec;
		return true;
	}
	return false;
}

__device__ __noinline__ bool BVHNode::Hit(const Ray& r, const Float tMin, Float tMax, HitRecord& rec) const
{
	Hittable* stack[10];
	int		  stack_ptr	   = 0;
	bool	  hit_anything = false;

	// Push root children (right first, then left)
	stack[stack_ptr++] = m_Right;
	stack[stack_ptr++] = m_Left;
	//HitRecord rec;

	while (stack_ptr > 0)
	{
		Hittable* node = stack[--stack_ptr];

		// Early out if bounding box doesn't hit.
		const AABB& box = node->GetBoundingBox(0.0, 1.0);
		if (!box.Hit(r, {tMin, tMax}))
			continue;

		if (node->IsLeaf())
		{
			if (processLeafNode(node, r, tMin, tMax, rec))
				hit_anything = true;
		}
		else
		{
			// Push right first, then left (so left is processed next)
			BVHNode* bvh_node  = static_cast<BVHNode*>(node);
			stack[stack_ptr++] = bvh_node->m_Right;
			stack[stack_ptr++] = bvh_node->m_Left;
		}
	}
	return hit_anything;
}
