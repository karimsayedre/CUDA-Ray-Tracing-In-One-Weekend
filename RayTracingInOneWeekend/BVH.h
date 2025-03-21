#pragma once
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"

struct BVHSoA
{
	uint32_t* m_left;  // Index of left child (or sphere index for leaves)
	uint32_t* m_right; // Index of right child (unused for leaves)
	Vec3* m_bounds_min;
	Vec3* m_bounds_max;
	bool*	   m_is_leaf;
	uint32_t   m_capacity;
	uint32_t   m_count;
	uint32_t   root;

	// Device constructor (not usable from host)
	__host__ __device__ BVHSoA(uint32_t max_nodes)
		: m_left(nullptr), m_right(nullptr), m_bounds_min(nullptr), m_bounds_max(nullptr), m_is_leaf(nullptr), m_capacity(max_nodes), m_count(0), root(0)
	{
	}

	// Host function to initialize device memory
	__host__ inline static void Init(BVHSoA*& d_bvh, uint32_t maxNodes)
	{
		BVHSoA h_bvh(maxNodes); // Temporary host instance

		// Allocate memory for arrays on the device
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_left, maxNodes * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_right, maxNodes * sizeof(uint32_t)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds_min, maxNodes * sizeof(Vec3)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds_max, maxNodes * sizeof(Vec3)));
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

	__device__ bool IntersectBounds(const Ray& ray, uint32_t node_index, Float t_min, Float& t_max) const
	{
		const Vec3	 min_bound(m_bounds_min[node_index]);
		const Vec3	 max_bound(m_bounds_max[node_index]);
		const Vec3& origin	 = ray.Origin();
		const Vec3& inv_dir = ray.InverseDirection();

		Vec3 t1 = (min_bound - origin) * inv_dir;
		Vec3 t2 = (max_bound - origin) * inv_dir;

		Vec3 tmin = glm::min(t1, t2);
		Vec3 tmax = glm::max(t1, t2);

		Float t_enter = glm::hmax(glm::hmax(tmin.x, tmin.y), tmin.z);
		Float t_exit  = glm::hmin(glm::hmin(tmax.x, tmax.y), tmax.z);

		return (t_exit > t_enter) && (t_enter < t_max) && (t_exit > t_min);
	}

	__device__ uint32_t BuildBVH_SoA(
		const HittableList* list,
		uint32_t*			indices,
		uint32_t			start,
		uint32_t			end);

	__device__ bool TraverseBVH_SoA(
		const Ray&	  ray,
		Float		  tmin,
		Float		  tmax,
		HittableList* list,
		uint32_t	  root_index,
		HitRecord&	  best_hit) const;
};
