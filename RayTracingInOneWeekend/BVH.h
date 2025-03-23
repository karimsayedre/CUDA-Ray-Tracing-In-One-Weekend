#pragma once
#include "HittableList.h"
#include "Ray.h"

struct alignas(32) BVHSoA
{
	struct alignas(4) BVH
	{
		uint16_t Left;	// Index of left child (or sphere index for leaves)
		uint16_t Right; // Index of right child (unused for leaves)
	};
	BVH* m_BVHs;

	AABB*	 m_Bounds;
	uint16_t m_Count = 0;
	uint16_t m_Root	 = 0;

	// Host function to initialize device memory
	__host__ static void Init(BVHSoA*& d_bvh, uint16_t maxNodes)
	{
		BVHSoA h_BVH; // Temporary host instance

		CHECK_CUDA_ERRORS(cudaMalloc(&h_BVH.m_Bounds, maxNodes * sizeof(AABB)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_BVH.m_BVHs, maxNodes * sizeof(BVH)));
		CHECK_CUDA_ERRORS(cudaMalloc(&d_bvh, sizeof(BVHSoA)));

		// Copy initialized BVH data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_bvh, &h_BVH, sizeof(BVHSoA), cudaMemcpyHostToDevice));
	}

	__device__ uint16_t AddNode(uint16_t left_idx, uint16_t right_idx, const AABB& box)
	{
		m_BVHs[m_Count]	  = BVH {left_idx, right_idx};
		m_Bounds[m_Count] = box;
		return m_Count++;
	}

	__device__ static float IntersectBounds(const AABB& bounds, const Vec3& origin, const Vec3& invDir, const int* dirIsNeg, float tmin, float tmax);

	__device__ uint16_t Build(const HittableList* list, uint16_t* indices, uint16_t start, uint16_t end);


	__device__ bool Traverse(const Ray& ray, float tmin, float tmax, HittableList* __restrict__ list, HitRecord& best_hit) const
	{
		// Use registers for stack instead of memory
		uint16_t currentNode = m_Root;
		uint16_t stackData[16];
		int		 stackPtr	  = 0;
		bool	 hit_anything = false;

		// Pre-compute ray inverse direction once
		const Vec3 invDir = 1.0f / ray.Direction();

		const int dirIsNeg[3] = {
			cuda::std::signbit(invDir.x),
			cuda::std::signbit(invDir.y),
			cuda::std::signbit(invDir.z),
		};

		stackData[stackPtr++] = currentNode;

		// Precompute common values once (stored in registers)
		const float ox = ray.Origin().x, oy = ray.Origin().y, oz = ray.Origin().z;
		const float idx = invDir.x, idy = invDir.y, idz = invDir.z;

		// Front-to-back traversal for early termination
		while (stackPtr != 0)
		{
			assert(stackPtr < 16);

			const BVH& node = m_BVHs[currentNode];

			// Process leaf node
			if (node.Right == UINT16_MAX)
			{
				// Process hit test
				hit_anything |= list->Hit(ray, tmin, tmax, best_hit, node.Left);

				currentNode = stackData[--stackPtr];
				continue;
			}

			// Check left child
			const AABB& leftBounds = m_Bounds[node.Left];
			float		tx1 = (leftBounds.Min.x - ox) * idx, tx2 = (leftBounds.Max.x - ox) * idx;
			float		ty1 = (leftBounds.Min.y - oy) * idy, ty2 = (leftBounds.Max.y - oy) * idy;
			float		tz1 = (leftBounds.Min.z - oz) * idz, tz2 = (leftBounds.Max.z - oz) * idz;
			float		tEnterL	 = fmaxf(fmaxf(dirIsNeg[0] ? tx2 : tx1, dirIsNeg[1] ? ty2 : ty1), fmaxf(dirIsNeg[2] ? tz2 : tz1, tmin));
			float		tExitL	 = fminf(fminf(dirIsNeg[0] ? tx1 : tx2, dirIsNeg[1] ? ty1 : ty2), fminf(dirIsNeg[2] ? tz1 : tz2, tmax));
			const float distLeft = (tEnterL > tExitL) ? FLT_MAX : tEnterL;

			// Check right child
			const AABB& rightBounds = m_Bounds[node.Right];
			tx1 = (rightBounds.Min.x - ox) * idx, tx2 = (rightBounds.Max.x - ox) * idx;
			ty1 = (rightBounds.Min.y - oy) * idy, ty2 = (rightBounds.Max.y - oy) * idy;
			tz1 = (rightBounds.Min.z - oz) * idz, tz2 = (rightBounds.Max.z - oz) * idz;
			float		tEnterR	  = fmaxf(fmaxf(dirIsNeg[0] ? tx2 : tx1, dirIsNeg[1] ? ty2 : ty1), fmaxf(dirIsNeg[2] ? tz2 : tz1, tmin));
			float		tExitR	  = fminf(fminf(dirIsNeg[0] ? tx1 : tx2, dirIsNeg[1] ? ty1 : ty2), fminf(dirIsNeg[2] ? tz1 : tz2, tmax));
			const float distRight = (tEnterR > tExitR) ? FLT_MAX : tEnterR;

			// Neither child was hit
			if (distLeft == FLT_MAX && distRight == FLT_MAX)
			{
				currentNode = stackData[--stackPtr];
				continue;
			}

			// Both children hit - traverse closer one first
			if (distLeft != FLT_MAX && distRight != FLT_MAX)
			{
				if (distLeft <= distRight)
				{
					// Left is closer
					currentNode			  = node.Left;
					stackData[stackPtr++] = node.Right;
				}
				else
				{
					// Right is closer
					currentNode			  = node.Right;
					stackData[stackPtr++] = node.Left;
				}

				continue;
			}

			// Only one child was hit
			currentNode = (distLeft != FLT_MAX) ? node.Left : node.Right;
		}

		return hit_anything;
	}
};
