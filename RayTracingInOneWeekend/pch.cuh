#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <device_launch_parameters.h>
#include <sm_60_atomic_functions.h> // For Pascal and newer GPUs

// #define GLM_CONFIG_SIMD GLM_ENABLE
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <thrust/sort.h>

#include <cuda_fp16.h>
#include "surface_functions.h"
#include <cuda_surface_types.h>
#include <math.h>


#if 0
using Float				 = HALF;
#define __float2half __float2half
#define HALF 1
#define hmin hmin
#define hmax hmax
#else
using Float				 = float;
#define __float2half
#define HALF 0
#define hmin min
#define hmax max
#endif

constexpr Float Infinity = std::numeric_limits<Float>::infinity();
inline Float	Pi		 = __float2half(3.1415926535897932385f);
// #define CUDART_INF_F (float)(0x7f800000)

constexpr float CUDART_INF_F = std::numeric_limits<float>::infinity();

using Vec3 = glm::vec<3, Float, glm::packed_mediump>;

// Boolean multiplication operators for branchless selection

// Vec3 * bool (select vector based on condition)
__device__ __forceinline__ Vec3 operator*(const Vec3& v, const bool condition)
{
	// Convert bool to Float (0.0 or 1.0)
	// Using __float2half to convert to half precision if needed
	Float factor = condition ? __float2half(1.0f) : __float2half(0.0f);
	return Vec3(v.x * factor, v.y * factor, v.z * factor);
}

// bool * Vec3 (commutative version)
__device__ __forceinline__ Vec3 operator*(const bool condition, const Vec3& v)
{
	Float factor = condition ? __float2half(1.0f) : __float2half(0.0f);
	return Vec3(v.x * factor, v.y * factor, v.z * factor);
}

//// Float * bool (scalar selection)
//__device__ __forceinline__ Float operator*(const Float value, const bool condition)
//{
//	return condition ? value : __float2half(0.0f);
//}
//
//// bool * Float (commutative version)
//__device__ __forceinline__ Float operator*(const bool condition, const Float value)
//{
//	return condition ? value : __float2half(0.0f);
//}

// Faster version for selection using direct bit manipulation (if needed)
__device__ __forceinline__ Vec3 selectVec3(const Vec3& v, const bool condition)
{
	// Create a mask from the boolean (all 1s or all 0s)
	// This assumes 16-bit half precision floats
	const uint16_t mask = condition ? 0xFFFF : 0x0000;

	// Use bitwise AND to apply the mask
	return Vec3(
		__half_as_ushort(v.x) & mask ? v.x : __float2half(0.0f),
		__half_as_ushort(v.y) & mask ? v.y : __float2half(0.0f),
		__half_as_ushort(v.z) & mask ? v.z : __float2half(0.0f));
}

// Vec3 selection between two vectors based on condition
__device__ __forceinline__ Vec3 selectVec3(const Vec3& whenTrue, const Vec3& whenFalse, const bool condition)
{
	return condition ? whenTrue : whenFalse;
}

// Float selection between two values based on condition
__device__ __forceinline__ Float selectFloat(const Float whenTrue, const Float whenFalse, const bool condition)
{
	return condition ? whenTrue : whenFalse;
}

namespace glm
{
#if HALF

	
	// Custom half-precision sqrt: converts __half -> float -> sqrt -> __half
	__device__ __host__ inline __half sqrt(__half x)
	{
		return __float2half(sqrtf(__half2float(x)));
	}

	__device__ __host__ inline __half hmin(__half a, __half b)
	{
		return (a < b) ? a : b;
	}

	__device__ __host__ inline __half hmax(__half a, __half b)
	{
		return (a > b) ? a : b;
	}

	// Component-wise minimum of two Vec3's
	__device__ __host__ inline Vec3 min(const Vec3& a, const Vec3& b)
	{
		return Vec3(
			glm::hmin(a.x, b.x),
			glm::hmin(a.y, b.y),
			glm::hmin(a.z, b.z));
	}

	// Component-wise maximum of two Vec3's
	__device__ __host__ inline Vec3 max(const Vec3& a, const Vec3& b)
	{
		return Vec3(
			glm::hmax(a.x, b.x),
			glm::hmax(a.y, b.y),
			glm::hmax(a.z, b.z));
	}

	// Component-wise square root of Vec3
	__device__ __host__ inline Vec3 sqrt(const Vec3& v)
	{
		return Vec3(
			__float2half(sqrtf((v.x))),
			__float2half(sqrtf((v.y))),
			__float2half(sqrtf((v.z))));
	}

	// Operator for adding two Vec3 objects
	__device__ __host__ inline Vec3 operator+(const Vec3& a, const Vec3& b)
	{
		return Vec3(__hadd(a.x, b.x), __hadd(a.y, b.y), __hadd(a.z, b.z));
	}

	// Operator for subtracting two Vec3 objects
	__device__ __host__ inline Vec3 operator-(const Vec3& a, const Vec3& b)
	{
		return Vec3(__hsub(a.x, b.x), __hsub(a.y, b.y), __hsub(a.z, b.z));
	}

	// Operator for multiplying two Vec3 objects
	__device__ __host__ inline Vec3 operator*(const Vec3& a, const Vec3& b)
	{
		return Vec3(__hmul(a.x, b.x), __hmul(a.y, b.y), __hmul(a.z, b.z));
	}

	// Scalar multiplication (Vec3 * __half)
	__device__ __host__ inline Vec3 operator*(const Vec3& a, const __half scalar)
	{
		return Vec3(__hmul(a.x, scalar), __hmul(a.y, scalar), __hmul(a.z, scalar));
	}

	// Scalar multiplication (__half * Vec3)
	__device__ __host__ inline Vec3 operator*(const __half scalar, const Vec3& a)
	{
		return Vec3(__hmul(scalar, a.x), __hmul(scalar, a.y), __hmul(scalar, a.z));
	}

	// Dot product (Vec3 . Vec3)
	__device__ __host__ inline __half dot(const Vec3& a, const Vec3& b)
	{
		return __hadd(__hmul(a.x, b.x), __hadd(__hmul(a.y, b.y), __hmul(a.z, b.z)));
	}

	// Cross product (Vec3 x Vec3)
	__device__ __host__ inline Vec3 cross(const Vec3& a, const Vec3& b)
	{
		return Vec3(
			__hsub(__hmul(a.y, b.z), __hmul(a.z, b.y)),
			__hsub(__hmul(a.z, b.x), __hmul(a.x, b.z)),
			__hsub(__hmul(a.x, b.y), __hmul(a.y, b.x)));
	}

	// Length (magnitude)
	__device__ __host__ inline auto length(const Vec3& v)
	{
		return glm::sqrt(dot(v, v));
	}

	// Normalize the vector
	__device__ __host__ inline Vec3 normalize(const Vec3& v)
	{
		auto len = glm::length(v);
		return v / len;
		//return Vec3(__hdiv(v.x, len), __hdiv(v.y, len), __hdiv(v.z, len));
	}

	// Scalar multiplication (float * Vec3)
	__device__ __host__ inline Vec3 operator*(const float scalar, const Vec3& v)
	{
		// Convert the scalar to __half and multiply each component of the Vec3 by it
		__half scalar_half = __float2half(scalar); // Convert scalar to __half
		return Vec3(__hmul(v.x, scalar_half),
					__hmul(v.y, scalar_half),
					__hmul(v.z, scalar_half));
	}

	__device__ __host__ inline Vec3 operator/(const float scalar, const Vec3& v)
	{
		return Vec3(__hdiv(__float2half(scalar), v.x),
					__hdiv(__float2half(scalar), v.y),
					__hdiv(__float2half(scalar), v.z));
	}

	__device__ __host__ inline __half operator/(const float scalar, const half v)
	{
		return __hdiv(__float2half(scalar), v);
	}

	__device__ __host__ inline __half operator*(const float a, const __half b)
	{
		return __hmul(__float2half(a), b);
	}

		// Spaceship operator (<=>) for __half
	__device__ __host__ inline std::strong_ordering operator<=>(const __half& lhs, const __half& rhs)
	{
		// Compare the two __half values
		if ((lhs) < (rhs))
			return std::strong_ordering::less;
		if (lhs > (rhs))
			return std::strong_ordering::greater;

		return std::strong_ordering::equal;
	}

	__device__ inline auto operator<=>(const Vec3& lhs, const Vec3& rhs)
	{
		if (auto cmp = lhs.x <=> rhs.x; cmp != 0)
			return cmp;
		if (auto cmp = lhs.y <=> rhs.y; cmp != 0)
			return cmp;
		return lhs.z <=> rhs.z;
	}



#endif
} // namespace glm
