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
	glm::vec3* m_bounds_min;
	glm::vec3* m_bounds_max;
	bool*	   m_is_leaf;
	uint32_t   m_capacity;
	uint32_t   m_count;
	uint32_t   root;

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
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds_min, maxNodes * sizeof(glm::vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds_max, maxNodes * sizeof(glm::vec3)));
		// CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds, maxNodes * sizeof(AABB)));
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
		// m_bounds[m_count]  = box;
		m_bounds_min[m_count] = box.Min;
		m_bounds_max[m_count] = box.Max;
		m_left[m_count]		  = left_idx;
		m_right[m_count]	  = right_idx;
		m_is_leaf[m_count]	  = leaf;
		return m_count++;
	}

	__device__ bool IntersectBounds(const Ray& ray, uint32_t node_index, float t_min, float& t_max) const
	{
		const glm::vec3	 min_bound(m_bounds_min[node_index]);
		const glm::vec3	 max_bound(m_bounds_max[node_index]);
		const glm::vec3& origin	 = ray.Origin();
		const glm::vec3& inv_dir = ray.InverseDirection();

		glm::vec3 t1 = (min_bound - origin) * inv_dir;
		glm::vec3 t2 = (max_bound - origin) * inv_dir;

		glm::vec3 tmin = glm::min(t1, t2);
		glm::vec3 tmax = glm::max(t1, t2);

		float t_enter = fmax(fmax(tmin.x, tmin.y), tmin.z);
		float t_exit  = fmin(fmin(tmax.x, tmax.y), tmax.z);

		return (t_exit > t_enter) && (t_enter < t_max) && (t_exit > t_min);
	}

	__device__ __forceinline__ uint32_t GetSplitAxis(uint32_t node_idx)
	{
		// Derive from bounds - determine which axis has the largest extent
		glm::vec3 min_bounds = (m_bounds_min[node_idx]);
		glm::vec3 max_bounds = (m_bounds_max[node_idx]);
		glm::vec3 extents	  = glm::vec3(
			   max_bounds.x - min_bounds.x,
			   max_bounds.y - min_bounds.y,
			   max_bounds.z - min_bounds.z);

		// Return axis with largest extent
		if (extents.x > extents.y && extents.x > extents.z)
			return 0;
		if (extents.y > extents.z)
			return 1;
		return 2;
	}

	__device__ uint32_t BuildBVH_SoA(
		const HittableList* list,
		uint32_t*			indices,
		uint32_t			start,
		uint32_t			end);

	__device__ bool TraverseBVH_SoA(
		const Ray&	  ray,
		float		  tmin,
		float		  tmax,
		HittableList* list,
		uint32_t	  root_index,
		HitRecord&	  best_hit);
};
