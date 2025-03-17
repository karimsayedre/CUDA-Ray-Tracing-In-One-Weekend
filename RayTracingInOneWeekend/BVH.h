#pragma once
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"

class Sphere;
//__device__ inline bool box_compare(const Hittable* a, const Hittable* b, int axis)
//{
//	return a->GetBoundingBox(0.0, 1.0).axis(axis).Min < b->GetBoundingBox(0.0, 1.0).axis(axis).Min;
//}
//
//__device__ inline bool box_x_compare(const Hittable* a, const Hittable* b)
//{
//	return box_compare(a, b, 0);
//}
//
//__device__ inline bool box_y_compare(const Hittable* a, const Hittable* b)
//{
//	return box_compare(a, b, 1);
//}
//
//__device__ inline bool box_z_compare(const Hittable* a, const Hittable* b)
//{
//	return box_compare(a, b, 2);
//}

struct BVHSoA
{
	uint32_t* m_left;  // Index of left child (or sphere index for leaves)
	uint32_t* m_right; // Index of right child (unused for leaves)
	AABB*	  m_bounds;
	bool*	  m_is_leaf;
	uint32_t  m_capacity;
	uint32_t  m_count;
	uint32_t  root;

	// Device constructor (not usable from host)
	__host__ __device__ BVHSoA(uint32_t max_nodes)
		: m_capacity(max_nodes), m_count(0), root(0)
	{
	}

	// Host function to initialize device memory
	__host__ inline static void Init(BVHSoA*& d_bvh, uint32_t maxNodes)
	{
		BVHSoA h_bvh(maxNodes); // Temporary host instance

		// Allocate memory for arrays on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_left, maxNodes * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_right, maxNodes * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds, maxNodes * sizeof(AABB)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_is_leaf, maxNodes * sizeof(bool)));

		h_bvh.m_capacity = maxNodes;
		h_bvh.m_count	 = 0;
		h_bvh.root		 = 0;

		// Allocate memory for BVH structure on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&d_bvh, sizeof(BVHSoA)));

		// Copy initialized BVH data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_bvh, &h_bvh, sizeof(BVHSoA), cudaMemcpyHostToDevice));
	}

	//__device__ ~BVHSoA()
	//{
	//	delete[] left;
	//	delete[] right;
	//	delete[] bounds;
	//	delete[] is_leaf;
	//}

	__device__ uint32_t AddNode(
		uint32_t	left_idx,
		uint32_t	right_idx,
		const AABB& box,
		bool		leaf)
	{
		m_bounds[m_count]  = box;
		m_left[m_count]	   = left_idx;
		m_right[m_count]   = right_idx;
		m_is_leaf[m_count] = leaf;
		return m_count++;
	}

};

__device__ uint32_t BuildBVH_SoA(
	const HittableList* list,
	uint32_t*			indices,
	uint32_t			start,
	uint32_t			end,
	BVHSoA*				soa);

__device__ bool TraverseBVH_SoA(
	const Ray&	  ray,
	float		  tmin,
	float		  tmax,
	HittableList* list,
	BVHSoA*		  soa,
	uint32_t	  root_index,
	HitRecord&	  best_hit);