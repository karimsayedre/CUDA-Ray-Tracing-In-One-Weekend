
#pragma once

#include "Renderer.h"
#include "BVH.h"
#include "Debug.h"
#include "Random.h"
#include "Material.h"

__device__ __host__ CPU_ONLY_INLINE void CreateWorld()
{
	const RenderParams* __restrict__ params = GetParams();
	uint32_t seed							= PcgHash(134537);

	// Ground sphere:
	Mat::Add(Mat::MaterialType::Lambert, Vec3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f);
	Hitables::Add(Vec3(0, -1000.0f, -1), 1000.0f);

	// For each grid position:
	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			// NOTE: explicitly having random variables to prevent different compilers from
			// having un-sequenced side effects so CPU and GPU versions are identical

			float chooseMat = RandomFloat(seed);

			// 2. draw the two offsets for the sphere center
			const float offX = RandomFloat(seed);
			const float offZ = RandomFloat(seed);
			Vec3		center(a + offX, 0.2f, b + offZ);

			if (chooseMat < 0.8f)
			{
				// Lambert: draw three components, square each
				const float r = RandomFloat(seed);
				const float g = RandomFloat(seed);
				const float b = RandomFloat(seed);
				Vec3		albedo(r * r, g * g, b * b);
				Mat::Add(Mat::MaterialType::Lambert, albedo, 0.0f, 1.0f);
			}
			else if (chooseMat < 0.95f)
			{
				// Metal: draw three components for color + one for fuzz
				const float mr = RandomFloat(seed);
				const float mg = RandomFloat(seed);
				const float mb = RandomFloat(seed);
				Vec3		metalCol(
					   0.5f * (1.0f + mr),
					   0.5f * (1.0f + mg),
					   0.5f * (1.0f + mb));
				const float fuzz = 0.5f * RandomFloat(seed);
				Mat::Add(Mat::MaterialType::Metal, metalCol, fuzz, 1.0f);
			}
			else
			{
				// Dielectric needs no extra RNG draws
				Mat::Add(Mat::MaterialType::Dielectric, Vec3(1.0f), 0.0f, 1.5f);
			}

			Hitables::Add(center, 0.2f);
		}
	}
	// Add the three big spheres:
	Mat::Add(Mat::MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
	Hitables::Add(Vec3(0, 1, 0), 1.0f);

	Mat::Add(Mat::MaterialType::Lambert, Vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f);
	Hitables::Add(Vec3(-4, 1, 0), 1.0f);

	Mat::Add(Mat::MaterialType::Metal, Vec3(0.7f, 0.6f, 0.5f), 0.0f, 1.0f);
	Hitables::Add(Vec3(4, 1, 0), 1.0f);

	uint32_t* indices = static_cast<uint32_t*>(malloc(params->List->Count * sizeof(uint32_t)));
	for (uint32_t index = 0; index < params->List->Count; ++index)
		indices[index] = index;

	params->BVH->m_Root = BVH::Build(indices, 0, params->List->Count);

#ifdef RTIOW_BVH_VEB
	params->BVH->m_Root = BVH::ReorderBVH(); // Disabled for now
#endif

	free(indices);

#ifdef __CUDA_ARCH__
	printf("CUDA BVH constructed with %u nodes.\n", params->BVH->m_Count);
	printf("CUDA BVH Root: %u\n", params->BVH->m_Root);
#else
	printf("CPU BVH constructed with %u nodes.\n", params->BVH->m_Count);
	printf("CPU BVH Root: %u\n", params->BVH->m_Root);
#endif

#ifdef RTIOW_DEBUG_BVH
	DebugBVHNode(params->BVH, params->BVH->m_Root);
#endif
}

[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE Vec3 RayColor(const Ray& ray, uint32_t& randSeed)
{
	const RenderParams* __restrict__ params = GetParams();

	Ray	 origRay = ray;
	Vec3 curAttenuation(1.0f);

	for (uint32_t i = 0; i < params->m_MaxDepth; i++)
	{
		HitRecord rec;
		// Early exit with sky color if no hit
		if (!BVH::Traverse(origRay, 0.001f, FLT_MAX, rec))
		{
			// Streamlined sky color calculation
			const float t = 0.5f * (origRay.Direction.y + 1.0f);
			return curAttenuation * ((1.0f - t) + t * Vec3(0.5f, 0.7f, 1.0f));
		}

		// if (i > 3)
		{
			const float rrProb = fmaxf(curAttenuation.x, fmaxf(curAttenuation.y, curAttenuation.z));

			// Early termination with zero - saves registers by avoiding division
			if (RandomFloat(randSeed) > rrProb)
				return Vec3(0.0f);

			// Apply RR adjustment directly to current attenuation
			curAttenuation /= rrProb;
		}

		// Early exit on no scatter
		if (!Mat::Scatter(origRay, rec, curAttenuation, randSeed))
			return Vec3(0.0f);
	}

	return curAttenuation; // Exceeded max depth
}
