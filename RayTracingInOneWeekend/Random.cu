#include "pch.cuh"

#include "Random.h"


__device__ uint32_t pcg_hash(uint32_t input)
{
	uint32_t state = input * 747796405u + 2891336453u;
	uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ float RandomFloat(uint32_t& seed, const float min, const float max)
{
	seed = pcg_hash(seed);
	return min + (float)(seed) / (float)(UINT_MAX / (max - min));
}

//__device__ int RandomInt(uint32_t& seed, const int min, const int max)
//{
//	return static_cast<int>(RandomFloat(seed, (float)min, (float)max + 1.0f));
//}




__device__ vec3 RandomVec3(uint32_t& seed, const float min, const float max)
{
	return { RandomFloat(seed, min, max), RandomFloat(seed, min, max), RandomFloat(seed, min, max) };
}

__device__ vec3 RandomNormalizedVector(uint32_t& seed)
{
	return (RandomVec3(seed).make_unit_vector());
}
