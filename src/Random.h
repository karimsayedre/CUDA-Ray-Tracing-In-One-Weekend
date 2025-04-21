#pragma once

__device__ __host__ __forceinline__ uint32_t PcgHash(const uint32_t input)
{
	const uint32_t state = input * 747796405u + 2891336453u;
	const uint32_t word	 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ __host__ __forceinline__ uint32_t RandomInt(uint32_t& seed)
{
	return (seed = (1664525 * seed + 1013904223));
}

__device__ __host__ __forceinline__ float RandomFloat(uint32_t& seed)
{
	//// float version using bitmask from Numerical Recipes
	// const uint one = 0x3f800000;
	// const uint msk = 0x007fffff;
	// return uintBitsToFloat(one | (msk & (RandomInt(seed) >> 9))) - 1;

	// Faster version from NVIDIA examples; quality good enough for our use case.
	return static_cast<float>(RandomInt(seed) & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

__device__ __host__ __forceinline__ Vec3 RandomVec3(uint32_t& seed)
{
	return {RandomFloat(seed), RandomFloat(seed), RandomFloat(seed)};
}
