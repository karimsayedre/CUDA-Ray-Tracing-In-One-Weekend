#pragma once
#include <cuda_runtime.h>
#include <numbers>
#include "Vec3.h"
#include <glm/geometric.hpp>

#include "CudaCamera.cuh"

//__device__ int RandomInt(uint32_t& seed, const int Min, const int Max);

__device__ inline int RandomInt(curandState* state, int min, int max)
{
	return min + (int)(curand_uniform(state) * (max - min + 1));
}

// template <typename OStream, typename Float>
//__device__ inline constexpr OStream& operator<<(OStream& out, const vec3& v) noexcept
//{
//	return out << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
// }

__device__ inline uint32_t pcg_hash(uint32_t input)
{
	uint32_t state = input * 747796405u + 2891336453u;
	uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ inline Float RandomFloat(uint32_t& seed, const float min, const float max)
{
	seed = pcg_hash(seed);
	return min + (float)(seed) / (float)(UINT_MAX / (max - min));
}

__device__ inline uint32_t RandomInt(uint32_t& seed)
{
	// LCG values from Numerical Recipes
	return (seed = (1664525 * seed + 1013904223));
}

__device__ inline Float RandomFloat(uint32_t& seed)
{
	//// Float version using bitmask from Numerical Recipes
	// const uint one = 0x3f800000;
	// const uint msk = 0x007fffff;
	// return uintBitsToFloat(one | (msk & (RandomInt(seed) >> 9))) - 1;

	// Faster version from NVIDIA examples; quality good enough for our use case.
	return (float(RandomInt(seed) & 0x00FFFFFF) / float(0x01000000));
}

//__device__ int RandomInt(uint32_t& seed, const int Min, const int Max)
//{
//	return static_cast<int>(RandomFloat(seed, (float)Min, (float)Max + 1.0f));
//}

__device__ inline Vec3 RandomVec3(uint32_t& seed, const float min, const float max)
{
	return {RandomFloat(seed, min, max), RandomFloat(seed, min, max), RandomFloat(seed, min, max)};
}

__device__ inline Vec3 RandomVec3(uint32_t& seed)
{
	return {RandomFloat(seed), RandomFloat(seed), RandomFloat(seed)};
}

//__device__ inline vec3 RandomNormalizedVector(uint32_t& seed)
//{
//	return (RandomVec3(seed).make_unit_vector());
//}

__device__ inline Vec3 randomUnitVector(uint32_t& randSeed)
{
	// Generate a random direction uniformly on the unit sphere
	// (One possible approach: spherical coordinates)
	float u = RandomFloat(randSeed);
	float v = RandomFloat(randSeed);

	float phi = 2.f * std::numbers::pi_v<float> * u;
	float z   = 1.f - 2.f * v;      // Range [-1, 1]
	float r   = sqrtf(1.f - z * z); // Radius in xy-plane

	return {r * cosf(phi), r * sinf(phi), z};
}

// Additional Optimization: Faster Random Generation
__device__ inline Vec3 RandomUnitVec3(uint32_t& seed)
{
	// Use faster, more uniform distribution
	float z	  = __float2half(2.0f) * RandomFloat(seed) - __float2half(1.0f);
	float r	  = sqrtf(1.0f - z * z);
	float phi = __float2half(2.0f) * __float2half(3.14159f) * RandomFloat(seed);

	return Vec3(
		r * cosf(phi),
		r * sinf(phi),
		z);
}
