#pragma once

#include "BVHPool.h"
#include "Hittable.h"
#include "HittableList.h"

class Sphere;
class BVHPool;
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
	BVHNode() = default;

	__device__ __noinline__ BVHNode(
		Hittable**	 src_objects,
		size_t		 start,
		size_t		 end,
		double		 time0,
		double		 time1,
		curandState* local_rand_state,
		BVHPool*	 pool);

	// Add a constructor that places the node in existing memory
	__device__ __noinline__ void Initialize(Hittable** src_objects, size_t start, size_t end, double time0, double time1, curandState* local_rand_state, BVHPool* pool);

	// Helper to process a leaf node hit.
	__device__ static bool processLeafNode(
		Hittable* __restrict__ node,
		const Ray&	r,
		const Float tMin,
		Float&		tMax,
		HitRecord&	rec);

	__device__ bool Hit(
		const Ray&	r,
		const Float tMin,
		Float		tMax,
		HitRecord&	rec) const override;


	//~BVHNode() override
	//{
	//	delete m_Left;
	//	delete m_Right;
	//}

  private:
	Hittable* m_Left;
	Hittable* m_Right;

	friend __device__ BVHNode* CreateBVHFromPool(Hittable** src_objects, size_t start, size_t end, double time0, double time1, curandState* local_rand_state, BVHPool* pool);
};

