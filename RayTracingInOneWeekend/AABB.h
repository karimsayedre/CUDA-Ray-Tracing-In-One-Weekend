#pragma once
#include <cuda_runtime.h>
//#include <nmmintrin.h>
#include "Ray.h"



//inline void SwapValues(__m128& a, __m128& b, const __m128 condition) {
//	__m128 temp = b;
//	b = _mm_blendv_ps(b, a, condition);
//	a = _mm_blendv_ps(a, temp, condition);
//}

//__device__ inline float min(float a, float b)
//{
//	return a < b ? a : b;
//}
//
//__device__ inline float max(float a, float b)
//{
//	return a > b ? a : b;
//}

class AABB
{
public:
	__device__ vec3 Min() const { return m_Minimum; }
	__device__ vec3 Max() const { return m_Maximum; }

	__device__ AABB(const vec3& min, const vec3& max)
		: m_Minimum(min), m_Maximum(max)
	{
	}

	AABB() = default;


	__device__  bool Hit(const Ray& r, Float tMin, Float tMax) const noexcept;

private:
	vec3 m_Minimum;
	vec3 m_Maximum;
};

