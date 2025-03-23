#pragma once

__device__ inline uint32_t PcgHash(uint32_t input)
{
	uint32_t state = input * 747796405u + 2891336453u;
	uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ inline uint32_t RandomInt(uint32_t& seed)
{
	// LCG values from Numerical Recipes
	return (seed = (1664525 * seed + 1013904223));
}

__device__ inline float RandomFloat(uint32_t& seed)
{
	//// float version using bitmask from Numerical Recipes
	// const uint one = 0x3f800000;
	// const uint msk = 0x007fffff;
	// return uintBitsToFloat(one | (msk & (RandomInt(seed) >> 9))) - 1;

	// Faster version from NVIDIA examples; quality good enough for our use case.
	return (float(RandomInt(seed) & 0x00FFFFFF) / float(0x01000000));
}

__device__ inline Vec3 RandomVec3(uint32_t& seed)
{
	return {RandomFloat(seed), RandomFloat(seed), RandomFloat(seed)};
}
