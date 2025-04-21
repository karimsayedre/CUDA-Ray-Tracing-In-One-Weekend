#pragma once
#include "HittableList.h"
#include "Ray.h"

namespace BVH
{

	struct alignas(32) BVHSoA
	{
		struct BVHNode
		{
			uint16_t Left;	// Index of left child (or sphere index for leaves)
			uint16_t Right; // Index of right child (unused for leaves)
		};
		BVHNode* m_Nodes;

		AABB*	 m_Bounds;
		uint16_t m_Count = 0;
		uint16_t m_Root	 = 0;
	};

	// Host function to initialize device memory
	__host__ static BVHSoA* Init(const uint16_t maxNodes)
	{
		BVHSoA h_BVH; // Temporary host instance

		CHECK_CUDA_ERRORS(cudaMalloc(&h_BVH.m_Bounds, maxNodes * sizeof(AABB)));
		CHECK_CUDA_ERRORS(cudaMalloc(&h_BVH.m_Nodes, maxNodes * sizeof(BVHSoA::BVHNode)));

		BVHSoA* d_BVH;
		CHECK_CUDA_ERRORS(cudaMalloc(&d_BVH, sizeof(BVHSoA)));

		// Copy initialized BVHNode data to device
		CHECK_CUDA_ERRORS(cudaMemcpy(d_BVH, &h_BVH, sizeof(BVHSoA), cudaMemcpyHostToDevice));
		return d_BVH;
	}

	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__device__ uint16_t AddNode(const uint16_t leftIdx, const uint16_t rightIdx, const AABB& box)
	{
		d_Params.BVH->m_Nodes[d_Params.BVH->m_Count]  = BVHSoA::BVHNode {leftIdx, rightIdx};
		d_Params.BVH->m_Bounds[d_Params.BVH->m_Count] = box;
		return d_Params.BVH->m_Count++;
	}

#ifdef VEB // can be more significant if we have a lot of nodes that they don't fit L1 cache
	__device__ void ReorderVEB(uint16_t nodeIdx, uint16_t* nodeMap, BVHSoA::BVHNode* tempNodes, AABB* tempBounds, uint16_t& newIndex, int currentDepth, int treeHeight);

	// Helper function for recursion with depth limits
	__device__ void ReorderVEBRecursive(uint16_t nodeIdx, uint16_t* nodeMap, BVHSoA::BVHNode* tempNodes, AABB* tempBounds, uint16_t& newIndex, int currentDepth, int depthLimit, int treeHeight)
	{
		// Return if node is invalid
		if (nodeIdx == UINT16_MAX)
			return;

		// Get the current node
		const BVHSoA::BVHNode& node = d_Params.BVH->m_Nodes[nodeIdx];

		// For leaf nodes, just add them to the new array
		if (node.Right == UINT16_MAX)
		{
			// Store the mapping from old to new index
			nodeMap[nodeIdx] = newIndex;

			// Copy the node and bounds to their new positions
			tempNodes[newIndex]	 = node;
			tempBounds[newIndex] = d_Params.BVH->m_Bounds[nodeIdx];

			// Move to next index
			newIndex++;
			return;
		}

		// If we're at the depth limit, process this subtree using VEB
		if (currentDepth >= depthLimit)
		{
			ReorderVEB(nodeIdx, nodeMap, tempNodes, tempBounds, newIndex, currentDepth, treeHeight);
			return;
		}

		// Otherwise, first add the node itself
		nodeMap[nodeIdx]	 = newIndex;
		tempNodes[newIndex]	 = node;
		tempBounds[newIndex] = d_Params.BVH->m_Bounds[nodeIdx];
		newIndex++;

		// Then recurse on left and right children
		ReorderVEBRecursive(node.Left, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, depthLimit, treeHeight);
		ReorderVEBRecursive(node.Right, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, depthLimit, treeHeight);
	}

	// Recursive function to reorder the nodes in a van Emde Boas layout
	__device__ void ReorderVEB(uint16_t nodeIdx, uint16_t* nodeMap, BVHSoA::BVHNode* tempNodes, AABB* tempBounds, uint16_t& newIndex, int currentDepth, int treeHeight)
	{
		// Return if node is invalid
		if (nodeIdx == UINT16_MAX)
			return;

		// Get the current node
		const BVHSoA::BVHNode& node = d_Params.BVH->m_Nodes[nodeIdx];

		// For leaf nodes, just add them to the new array
		if (node.Right == UINT16_MAX)
		{
			// Store the mapping from old to new index
			nodeMap[nodeIdx] = newIndex;

			// Copy the node and bounds to their new positions
			tempNodes[newIndex]	 = node;
			tempBounds[newIndex] = d_Params.BVH->m_Bounds[nodeIdx];

			// Move to next index
			newIndex++;
			return;
		}

		// Calculate the height of the remaining subtree
		int height		= treeHeight - currentDepth;
		int upperHeight = height / 2;

		// For internal nodes, first add the node itself
		nodeMap[nodeIdx]	 = newIndex;
		tempNodes[newIndex]	 = node;
		tempBounds[newIndex] = d_Params.BVH->m_Bounds[nodeIdx];
		newIndex++;

		// Then recurse on top part of tree (upper levels)
		ReorderVEBRecursive(node.Left, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, currentDepth + upperHeight, treeHeight);

		// Then recurse on bottom part of tree (lower levels)
		ReorderVEBRecursive(node.Right, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, currentDepth + upperHeight, treeHeight);
	}

	__device__ uint16_t ReorderBVH()
	{
		// Temporary arrays to hold the reordered BVH
		BVHSoA::BVHNode* tempNodes	= static_cast<BVHSoA::BVHNode*>(malloc(d_Params.BVH->m_Count * sizeof(BVHSoA::BVHNode)));
		AABB*			 tempBounds = static_cast<AABB*>(malloc(d_Params.BVH->m_Count * sizeof(AABB)));

		// Create a map to store the old -> new node indices mapping
		uint16_t* nodeMap = static_cast<uint16_t*>(malloc(d_Params.BVH->m_Count * sizeof(uint16_t)));

		// Calculate the height of the tree (approximate for potentially unbalanced trees)
		// For a binary tree with n nodes, height is approximately log2(n)
		int treeHeight = static_cast<int>(log2f(static_cast<float>(d_Params.BVH->m_Count))) + 1;

		// First pass: Copy nodes to temp arrays in van Emde Boas layout
		uint16_t newIndex = 0;
		ReorderVEB(d_Params.BVH->m_Root, nodeMap, tempNodes, tempBounds, newIndex, 0, treeHeight);

		// Save the new root index
		uint16_t newRootIndex = nodeMap[d_Params.BVH->m_Root];

		// Second pass: Update child indices using the mapping
		for (uint16_t i = 0; i < d_Params.BVH->m_Count; ++i)
		{
			// If not a leaf node, update the child indices
			if (tempNodes[i].Right != UINT16_MAX)
			{
				tempNodes[i].Left  = nodeMap[tempNodes[i].Left];
				tempNodes[i].Right = nodeMap[tempNodes[i].Right];
			}
		}

		// Copy back the reordered data
		memcpy(d_Params.BVH->m_Nodes, tempNodes, d_Params.BVH->m_Count * sizeof(BVHSoA::BVHNode));
		memcpy(d_Params.BVH->m_Bounds, tempBounds, d_Params.BVH->m_Count * sizeof(AABB));

		// Clean up
		free(tempNodes);
		free(tempBounds);
		free(nodeMap);

		// Return the new root index
		return newRootIndex;
	}
#endif

	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__device__ uint16_t Build(uint16_t* indices, uint16_t start, uint16_t end)
	{
		uint16_t objectSpan = end - start;

		// Compute bounding box for this node
		AABB box;
		bool firstBox = true;
		for (uint16_t i = start; i < end; ++i)
		{
			uint32_t	sphereIndex = indices[i];
			const AABB& currentBox	= d_Params.List->AABBs[sphereIndex];
			if (firstBox)
			{
				box		 = currentBox;
				firstBox = false;
			}
			else
			{
				box = AABB(box, currentBox);
			}
		}

		// Handle leaf cases
		if (objectSpan == 1)
		{
			// Leaf node: store sphere index
			return AddNode(indices[start], UINT16_MAX, box);
		}

		if (objectSpan == 2)
		{
			uint16_t idxA = indices[start];
			uint16_t idxB = indices[start + 1];

			// Create leaf nodes for each sphere
			const AABB& boxA = d_Params.List->AABBs[idxA];
			const AABB& boxB = d_Params.List->AABBs[idxB];

			uint16_t leftLeaf  = AddNode(idxA, UINT16_MAX, boxA);
			uint16_t rightLeaf = AddNode(idxB, UINT16_MAX, boxB);

			// Create parent internal node
			const AABB combined(boxA, boxB);
			return AddNode(leftLeaf, rightLeaf, combined);
		}

		// Use SAH to find the best split
		float	 bestCost	  = FLT_MAX;
		uint16_t bestAxis	  = 0;
		uint16_t bestSplitIdx = start + objectSpan / 2; // Default middle split as fallback

		// Cost of not splitting (creating a leaf)
		// float no_split_cost = objectSpan * box.SurfaceArea();

		// Try each axis
		for (uint16_t axis = 0; axis < 3; ++axis)
		{
			// Sort indices along this axis
			auto comparator = [&](const uint32_t a, const uint32_t b)
			{
				return d_Params.List->AABBs[a].Center()[axis] < d_Params.List->AABBs[b].Center()[axis];
			};

			thrust::sort(indices + start, indices + end, comparator);

			// Precompute all bounding boxes from left to right
			AABB* leftBoxes = (AABB*)malloc(sizeof(AABB) * objectSpan);
			leftBoxes[0]	= d_Params.List->AABBs[indices[start]];
			for (uint16_t i = 1; i < objectSpan; ++i)
			{
				leftBoxes[i] = AABB(leftBoxes[i - 1], d_Params.List->AABBs[indices[start + i]]);
			}

			// Precompute all bounding boxes from right to left
			AABB* rightBoxes		   = (AABB*)malloc(sizeof(AABB) * objectSpan);
			rightBoxes[objectSpan - 1] = d_Params.List->AABBs[indices[end - 1]];
			for (int i = objectSpan - 2; i >= 0; --i)
			{
				rightBoxes[i] = AABB(rightBoxes[i + 1], d_Params.List->AABBs[indices[start + i]]);
			}

			// Evaluate SAH cost for each possible split
			for (uint16_t i = 1; i < objectSpan; ++i)
			{
				const float leftCount  = i;
				const float rightCount = (float)(objectSpan - i);

				const float leftSa	= leftBoxes[i - 1].SurfaceArea();
				const float rightSa = rightBoxes[i].SurfaceArea();

				// SAH cost formula: C = T_traverse + (left_count * left_sa + right_count * right_sa) / parent_sa * T_intersect
				// We can simplify by using constant traversal and intersection costs
				constexpr float traversalCost	 = 0.3f;
				constexpr float intersectionCost = 1.0f;
				const float		parentSA		 = box.SurfaceArea();
				const float		cost			 = traversalCost + (leftCount * leftSa + rightCount * rightSa) / parentSA * intersectionCost;
				if (cost < bestCost)
				{
					bestCost	 = cost;
					bestAxis	 = axis;
					bestSplitIdx = start + i;
				}
			}

			free(leftBoxes);
			free(rightBoxes);
		}

		// If no split is better than not splitting, and we have fewer than some threshold of objects,
		// we could make this a leaf. However, we'll always split for simplicity and compatibility
		// with the traversal function.

		// Resort along the best axis if it's not the last one we tried
		if (bestAxis != 2)
		{
			auto comparator = [&](const uint32_t a, uint32_t b)
			{
				return d_Params.List->AABBs[a].Center()[bestAxis] < d_Params.List->AABBs[b].Center()[bestAxis];
			};

			thrust::sort(indices + start, indices + end, comparator);
		}

		// Recursively build children
		const uint16_t leftIdx	= Build(indices, start, bestSplitIdx);
		const uint16_t rightIdx = Build(indices, bestSplitIdx, end);

		// Compute the combined bounding box from the actual child nodes
		AABB		combined  = d_Params.BVH->m_Bounds[leftIdx];
		const AABB& rightAabb = d_Params.BVH->m_Bounds[rightIdx];
		combined			  = AABB(combined, rightAabb);

		return AddNode(leftIdx, rightIdx, combined);
	}

	__device__ inline bool Traverse(const Ray& ray, const float tmin, float tmax, HitRecord& bestHit)
	{
		// Use registers for stack instead of memory
		uint16_t currentNode = d_Params.BVH->m_Root;
		uint16_t stackData[16];
		int		 stackPtr	 = 0;
		bool	 hitAnything = false;

		// Pre-compute ray inverse direction once
		const __align__(16) Vec3 invDir = 1.0f / ray.Direction;

		const bool dirIsNeg[3] = {
			signbit(invDir.x),
			signbit(invDir.y),
			signbit(invDir.z),
		};

		stackData[stackPtr++] = currentNode;

		// Front-to-back traversal for early termination
		while (stackPtr != 0)
		{
			assert(stackPtr < 16);

			const BVH::BVHSoA::BVHNode& node = d_Params.BVH->m_Nodes[currentNode];

			// Process leaf node
			if (node.Right == UINT16_MAX)
			{
				// Process hit test
				hitAnything |= Hitables::IntersectPrimitive(ray, tmin, tmax, bestHit, node.Left);

				currentNode = stackData[--stackPtr];
				continue;
			}

			// Check left child
			const AABB& leftBounds = d_Params.BVH->m_Bounds[node.Left];
			float		tx1 = (leftBounds.Min.x - ray.Origin.x) * invDir.x, tx2 = (leftBounds.Max.x - ray.Origin.x) * invDir.x;
			float		ty1 = (leftBounds.Min.y - ray.Origin.y) * invDir.y, ty2 = (leftBounds.Max.y - ray.Origin.y) * invDir.y;
			float		tz1 = (leftBounds.Min.z - ray.Origin.z) * invDir.z, tz2 = (leftBounds.Max.z - ray.Origin.z) * invDir.z;
			const float tEnterL	 = fmaxf(fmaxf(dirIsNeg[0] ? tx2 : tx1, dirIsNeg[1] ? ty2 : ty1), fmaxf(dirIsNeg[2] ? tz2 : tz1, tmin));
			const float tExitL	 = fminf(fminf(dirIsNeg[0] ? tx1 : tx2, dirIsNeg[1] ? ty1 : ty2), fminf(dirIsNeg[2] ? tz1 : tz2, tmax));
			const float distLeft = (tEnterL > tExitL) ? FLT_MAX : tEnterL;

			// Check right child
			const AABB& rightBounds = d_Params.BVH->m_Bounds[node.Right];
			tx1 = (rightBounds.Min.x - ray.Origin.x) * invDir.x, tx2 = (rightBounds.Max.x - ray.Origin.x) * invDir.x;
			ty1 = (rightBounds.Min.y - ray.Origin.y) * invDir.y, ty2 = (rightBounds.Max.y - ray.Origin.y) * invDir.y;
			tz1 = (rightBounds.Min.z - ray.Origin.z) * invDir.z, tz2 = (rightBounds.Max.z - ray.Origin.z) * invDir.z;
			const float tEnterR	  = fmaxf(fmaxf(dirIsNeg[0] ? tx2 : tx1, dirIsNeg[1] ? ty2 : ty1), fmaxf(dirIsNeg[2] ? tz2 : tz1, tmin));
			const float tExitR	  = fminf(fminf(dirIsNeg[0] ? tx1 : tx2, dirIsNeg[1] ? ty1 : ty2), fminf(dirIsNeg[2] ? tz1 : tz2, tmax));
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

		return hitAnything;
	}
} // namespace BVH
