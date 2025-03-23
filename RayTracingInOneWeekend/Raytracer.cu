#include "pch.cuh"
#include "BVH.h"
#include "HittableList.h"
#include "Material.h"

__global__ void CreateWorld(HittableList* d_List, Materials* d_Materials, BVHSoA* d_World)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state;
		curand_init(1984, 0, 0, &local_rand_state);

		// Ground sphere:
		d_Materials->Add(MaterialType::Lambert, Vec3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f);
		d_List->Add(Vec3(0, -1000.0f, -1), 1000.0f);

		// For each grid position:
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
#define RND (curand_uniform(&local_rand_state))

				float choose_mat = RND;
				Vec3  center(a + RND, 0.2f, b + RND);
				if (choose_mat < 0.8f)
				{
					d_Materials->Add(MaterialType::Lambert, Vec3(RND * RND, RND * RND, RND * RND), 0.0f, 1.0f);
					d_List->Add(center, 0.2f);
				}
				else if (choose_mat < 0.95f)
				{
					d_Materials->Add(MaterialType::Metal, Vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND, 1.0f);
					d_List->Add(center, 0.2f);
				}
				else
				{
					d_Materials->Add(MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
					d_List->Add(center, 0.2f);
				}
#undef RND
			}
		}

		// Add the three big spheres:
		d_Materials->Add(MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
		d_List->Add(Vec3(0, 1, 0), 1.0f);

		d_Materials->Add(MaterialType::Lambert, Vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f);
		d_List->Add(Vec3(-4, 1, 0), 1.0f);

		d_Materials->Add(MaterialType::Metal, Vec3(0.7f, 0.6f, 0.5f), 0.0f, 1.0f);
		d_List->Add(Vec3(4, 1, 0), 1.0f);

		uint16_t* indices = (uint16_t*)malloc(d_List->GetPrimitiveCount() * sizeof(uint16_t));
		for (uint16_t index = 0; index < d_List->GetPrimitiveCount(); ++index)
			indices[index] = index;

		d_World->m_Root = d_World->Build(d_List, indices, 0, d_List->GetPrimitiveCount());
		printf("BVH created with %u nodes.\n", d_World->m_Count);

		printf("BVH Root: %u\n", d_World->m_Root);
		// DebugBVHNode(d_World, d_World->root);

		free(indices);
	}
}

__device__ Vec3 RayColor(Ray& ray, BVHSoA* __restrict__ d_World, HittableList* __restrict__ d_List, Materials* __restrict__ d_Materials, const uint32_t depth, uint32_t& randSeed)
{
	Vec3 curAttenuation(1.0f);

	#pragma unroll 1
	for (uint32_t i = 0; i < depth; i++)
	{
		HitRecord rec;
		// Early exit with sky color if no hit
		if (!d_World->Traverse(ray, 0.001f, FLT_MAX, d_List, rec))
		{
			// Streamlined sky color calculation
			const float t = (0.5f) * (ray.Direction().y + 1.0f);
			return curAttenuation * ((1.0f - t) * 1.0f + t * Vec3(0.5f, 0.7f, 1.0f));
		}

		// Russian Roulette - simplified
		// Only apply after a few bounces (commented out in original)
		/*if (i > 3)*/ 
			// Use max component for probability
			const float rrProb = glm::max(curAttenuation.x, glm::max(curAttenuation.y, curAttenuation.z));

			// Early termination with zero - saves registers by avoiding division
			if (RandomFloat(randSeed) > rrProb)
				return Vec3(0.0f);

			// Apply RR adjustment directly to current attenuation
			curAttenuation /= rrProb;
		

		// Early exit on no scatter
		if (!d_Materials->Scatter(ray, rec, curAttenuation, randSeed))
			return curAttenuation;
	}

	return Vec3(0.0f); // Exceeded max depth
}

// Modified kernel using surface object
__global__ void InternalRender(cudaSurfaceObject_t fb, BVHSoA* __restrict__ d_World, HittableList* __restrict__ d_List, Materials* __restrict__ d_Materials, uint32_t maxX, uint32_t maxY, Camera* camera, uint32_t samplersPerPixel, float colorMul, uint32_t maxDepth, uint32_t* randSeeds)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= maxX) || (y >= maxY))
		return;

	uint32_t pixelIndex = y * maxX + x;

	uint32_t seed = randSeeds[pixelIndex];
	Vec3	 pixelColor(0.0f);
	for (uint32_t s = 0; s < samplersPerPixel; s++)
	{
		float u	 = float(float(x) + RandomFloat(seed)) / float(maxX);
		float v	 = float(float(y) + RandomFloat(seed)) / float(maxY);
		v		 = 1.0f - v;
		auto ray = camera->GetRay(u, v);

		pixelColor += RayColor(ray, d_World, d_List, d_Materials, maxDepth, seed);
	}
	randSeeds[pixelIndex] = seed;
	pixelColor *= colorMul;
	pixelColor = glm::sqrt(pixelColor);

	// Convert to uchar4 format
	uchar4 pixel = make_uchar4(
		static_cast<unsigned char>(pixelColor.x * 255.f),
		static_cast<unsigned char>(pixelColor.y * 255.f),
		static_cast<unsigned char>(pixelColor.z * 255.f),
		255);

	// Write to surface
	surf2Dwrite(pixel, fb, x * sizeof(uchar4), y);
}
