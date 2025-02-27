#pragma once
#include <cuda_runtime.h>
#include "Vec3.h"
#include <glm/geometric.hpp>



__device__ uint32_t pcg_hash(uint32_t input);

__device__ float RandomFloat(uint32_t& seed, const float min = 0.0f, const float max = 1.0f);

//__device__ int RandomInt(uint32_t& seed, const int min, const int max);

__device__ inline int RandomInt(curandState* state, int min, int max)
{
	return min + (int)(curand_uniform(state) * (max - min + 1));
}

__device__ vec3 RandomVec3(uint32_t& seed, const float min = 0.0f, const float max = 1.0f);

//template <typename OStream, typename Float>
//__device__ inline constexpr OStream& operator<<(OStream& out, const vec3& v) noexcept
//{
//	return out << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
//}


__device__ vec3 RandomNormalizedVector(uint32_t& seed);
