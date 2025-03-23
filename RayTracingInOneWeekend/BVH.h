#pragma once
#include "HittableList.h"
#include "Ray.h"

struct alignas(32) BVHSoA
{
	struct alignas(8) BVH
	{
		uint16_t m_left;  // Index of left child (or sphere index for leaves)
		uint16_t m_right; // Index of right child (unused for leaves)
		uint16_t m_is_leaf;
	};

	// Vec3 m_bounds_min;
	// Vec3 m_bounds_max;
	AABB* m_bounds;

	BVH*	 m_BVHs;
	uint16_t m_count;
	uint16_t root;

	// Host function to initialize device memory
	__host__ inline static void Init(BVHSoA*& d_bvh, uint16_t maxNodes)
	{
		BVHSoA h_bvh; // Temporary host instance

		// Allocate memory for arrays on the device
		// CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_left, maxNodes * sizeof(uint16_t)));
		// CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_right, maxNodes * sizeof(uint16_t)));

		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds, maxNodes * sizeof(AABB)));
		//  CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds_min, maxNodes * sizeof(Vec3)));
		//  CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_bounds_max, maxNodes * sizeof(Vec3)));
		// CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_is_leaf, maxNodes * sizeof(uint16_t)));

		CHECK_CUDA_ERRORS(cudaMalloc(&h_bvh.m_BVHs, maxNodes * sizeof(BVH)));

		//h_bvh.m_capacity = maxNodes;
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

	__device__ uint16_t AddNode(
		uint16_t	left_idx,
		uint16_t	right_idx,
		const AABB& box,
		uint16_t	leaf)
	{
		// m_bounds[m_count] = box;
		//  m_bounds_min[m_count] = box.Min;
		//  m_bounds_max[m_count] = box.Max;
		// m_left[m_count]	   = left_idx;
		// m_right[m_count]   = right_idx;
		// m_is_leaf[m_count] = leaf;

		m_BVHs[m_count] = BVH {
			left_idx,
			right_idx,
			// box,
			leaf};

		m_bounds[m_count] = box;
		return m_count++;
	}

	// Helper to compute approximate distance to node center for traversal ordering
	__device__ float ComputeNodeDistance(const Ray& ray, uint16_t nodeIdx) const
	{
		// Compute center of node bounds

		Vec3 center = m_bounds[nodeIdx].Center();

		// Distance from ray origin to center
		Vec3 toCenter = center - ray.Origin();
		return dot(toCenter, ray.Direction());
	}

	__device__ bool IntersectBoundsFast(
		const Vec3& rayOrigin,
		const Vec3& invDir,
		const int*	dirIsNeg, // Array of 3 ints indicating if direction is negative
		uint16_t	node_index,
		Float		t_min,
		Float		t_max) const
	{
		const AABB& bounds = m_bounds[node_index];

		// Using dirIsNeg to select min/max bounds directly
		Float t_x1	  = (bounds.Min.x - rayOrigin.x) * invDir.x;
		Float t_x2	  = (bounds.Max.x - rayOrigin.x) * invDir.x;
		Float t_min_x = dirIsNeg[0] ? t_x2 : t_x1;
		Float t_max_x = dirIsNeg[0] ? t_x1 : t_x2;

		Float t_y1	  = (bounds.Min.y - rayOrigin.y) * invDir.y;
		Float t_y2	  = (bounds.Max.y - rayOrigin.y) * invDir.y;
		Float t_min_y = dirIsNeg[1] ? t_y2 : t_y1;
		Float t_max_y = dirIsNeg[1] ? t_y1 : t_y2;

		Float t_z1	  = (bounds.Min.z - rayOrigin.z) * invDir.z;
		Float t_z2	  = (bounds.Max.z - rayOrigin.z) * invDir.z;
		Float t_min_z = dirIsNeg[2] ? t_z2 : t_z1;
		Float t_max_z = dirIsNeg[2] ? t_z1 : t_z2;

		// Finding entry and exit points
		Float t_enter = glm::max(glm::max(t_min_x, t_min_y), glm::max(t_min_z, t_min));
		Float t_exit  = glm::min(glm::min(t_max_x, t_max_y), glm::min(t_max_z, t_max));

		return t_exit > t_enter;
	}

	__device__ uint16_t BuildBVH_SoA(
		const HittableList* list,
		uint16_t*			indices,
		uint16_t			start,
		uint16_t			end);

	__device__ bool TraverseBVH_SoA(
		const Ray&	  ray,
		Float		  tmin,
		Float		  tmax,
		HittableList* list,
		HitRecord&	  best_hit) const;
};
