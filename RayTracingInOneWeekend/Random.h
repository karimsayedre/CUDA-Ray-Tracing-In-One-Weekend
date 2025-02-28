#pragma once
#include <cuda_runtime.h>
#include "Vec3.h"
#include <glm/geometric.hpp>

//__device__ int RandomInt(uint32_t& seed, const int min, const int max);

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

__device__ inline float RandomFloat(uint32_t& seed, const float min, const float max)
{
	seed = pcg_hash(seed);
	return min + (float)(seed) / (float)(UINT_MAX / (max - min));
}

//__device__ int RandomInt(uint32_t& seed, const int min, const int max)
//{
//	return static_cast<int>(RandomFloat(seed, (float)min, (float)max + 1.0f));
//}

__device__ inline vec3 RandomVec3(uint32_t& seed, const float min = 0.0f, const float max = 1.0f)
{
	return {RandomFloat(seed, min, max), RandomFloat(seed, min, max), RandomFloat(seed, min, max)};
}

__device__ inline vec3 RandomNormalizedVector(uint32_t& seed)
{
	return (RandomVec3(seed).make_unit_vector());
}