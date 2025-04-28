
#pragma once

#include "Renderer.h"
#include "CPU_GPU.h"
#include "BVH.h"
#include "Random.h"
#include "Material.h"



template<typename RNG>
__device__ __host__ inline void CreateWorld()
{
	RNG			rng(1984); // Seed the RNG
	const auto* params = GetParams();

	// Ground sphere:
	Mat::Add(Mat::MaterialType::Lambert, Vec3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f);
	Hitables::Add(Vec3(0, -1000.0f, -1), 1000.0f);

	// For each grid position:
	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
#define RND (rng.Uniform())

			const float chooseMat = RND;
			const Vec3	center(a + RND, 0.2f, b + RND);
			if (chooseMat < 0.8f)
			{
				Mat::Add(Mat::MaterialType::Lambert, Vec3(RND * RND, RND * RND, RND * RND), 0.0f, 1.0f);
				Hitables::Add(center, 0.2f);
			}
			else if (chooseMat < 0.95f)
			{
				Mat::Add(Mat::MaterialType::Metal, Vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND, 1.0f);
				Hitables::Add(center, 0.2f);
			}
			else
			{
				Mat::Add(Mat::MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
				Hitables::Add(center, 0.2f);
			}
#undef RND
		}
	}

	// Add the three big spheres:
	Mat::Add(Mat::MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
	Hitables::Add(Vec3(0, 1, 0), 1.0f);

	Mat::Add(Mat::MaterialType::Lambert, Vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f);
	Hitables::Add(Vec3(-4, 1, 0), 1.0f);

	Mat::Add(Mat::MaterialType::Metal, Vec3(0.7f, 0.6f, 0.5f), 0.0f, 1.0f);
	Hitables::Add(Vec3(4, 1, 0), 1.0f);

	uint16_t* indices = static_cast<uint16_t*>(malloc(params->List->Count * sizeof(uint16_t)));
	for (uint16_t index = 0; index < params->List->Count; ++index)
		indices[index] = index;

	params->BVH->m_Root = BVH::Build(indices, 0, params->List->Count);

	// params->BVH->m_Root = BVH::ReorderBVH(); // Disabled for now

	free(indices);

	printf("BVHNode created with %u nodes.\n", params->BVH->m_Count);
	printf("BVHNode Root: %u\n", params->BVH->m_Root);
	// DebugBVHNode(d_World, d_World->root);
}



__device__ __host__ inline Vec3 RayColor(const Ray& ray, uint32_t& randSeed)
{
	const auto* params = GetParams();

	Ray	 origRay = ray;
	Vec3 curAttenuation(1.0f);

	for (uint32_t i = 0; i < params->m_MaxDepth; i++)
	{
		HitRecord rec;
		// Early exit with sky color if no hit
		if (!BVH::Traverse(origRay, 0.001f, FLT_MAX, rec))
		{
			// Streamlined sky color calculation
			const float t = (0.5f) * (origRay.Direction.y + 1.0f);
			return curAttenuation * ((1.0f - t) * 1.0f + t * Vec3(0.5f, 0.7f, 1.0f));
		}

		// if (i > 3)
		{
			const float rrProb = glm::max(curAttenuation.x, glm::max(curAttenuation.y, curAttenuation.z));

			// Early termination with zero - saves registers by avoiding division
			if (RandomFloat(randSeed) > rrProb)
				return Vec3(0.0f);

			// Apply RR adjustment directly to current attenuation
			curAttenuation /= rrProb;
		}

		// Early exit on no scatter
		if (!Mat::Scatter(origRay, rec, curAttenuation, randSeed))
			return curAttenuation;
	}

	return curAttenuation; // Exceeded max depth
}

