#pragma once
#include <limits>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <glm/gtx/fast_square_root.hpp>
#include <cmath>


using Float				 = float;
constexpr Float Infinity = std::numeric_limits<Float>::infinity();
constexpr Float Pi		 = 3.1415926535897932385f;
// #define CUDART_INF_F (float)(0x7f800000)

constexpr float CUDART_INF_F = std::numeric_limits<float>::infinity();

#if 0
class vec3
{
  public:
	__host__ __device__ vec3()
	{
	}
	__host__ __device__ vec3(float e0, float e1, float e2)
	{
		e.x = e0;
		e.y = e1;
		e.z = e2;
	}

	__host__ __device__ vec3(float e0)
	{
		e.x = e0;
		e.y = e0;
		e.z = e0;
	}

	__host__ __device__ inline float x() const
	{
		return e.x;
	}
	__host__ __device__ inline float y() const
	{
		return e.y;
	}
	__host__ __device__ inline float z() const
	{
		return e.z;
	}
	__host__ __device__ inline float r() const
	{
		return e.x;
	}
	__host__ __device__ inline float g() const
	{
		return e.y;
	}
	__host__ __device__ inline float b() const
	{
		return e.z;
	}

	__host__ __device__ inline const vec3& operator+() const
	{
		return *this;
	}
	__host__ __device__ inline vec3 operator-() const
	{
		return vec3(-e.x, -e.y, -e.z);
	}
	__host__ __device__ inline float operator[](int i) const noexcept
	{
		switch (i)
		{
			case 0: return e.x;
			case 1: return e.y;
			case 2: return e.z;
		}
		// assert(false);
		return 0;
	}
	__host__ __device__ inline float& operator[](int i) noexcept
	{
		switch (i)
		{
			case 0: return e.x;
			case 1: return e.y;
			case 2: return e.z;
		}
		// assert(false);
		return e.x;
	};

	__host__ __device__ inline vec3& operator+=(const vec3& v2);
	__host__ __device__ inline vec3& operator-=(const vec3& v2);
	__host__ __device__ inline vec3& operator*=(const vec3& v2);
	__host__ __device__ inline vec3& operator/=(const vec3& v2);
	__host__ __device__ inline vec3& operator*=(const float t);
	__host__ __device__ inline vec3& operator/=(const float t);

	__host__ __device__ inline float length() const
	{
		return sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
	}
	__host__ __device__ inline float squared_length() const
	{
		return e.x * e.x + e.y * e.y + e.z * e.z;
	}
	__host__ __device__ inline vec3 make_unit_vector();

	float3 e;
};

inline std::istream& operator>>(std::istream& is, vec3& t)
{
	is >> t.e.x >> t.e.y >> t.e.z;
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t)
{
	os << t.e.x << " " << t.e.y << " " << t.e.z;
	return os;
}

__host__ __device__ inline vec3 vec3::make_unit_vector()
{
	float k = 1.0f / sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
	e.x *= k;
	e.y *= k;
	e.z *= k;
	return {e.x, e.y, e.z};
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e.x + v2.e.x, v1.e.y + v2.e.y, v1.e.z + v2.e.z);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e.x - v2.e.x, v1.e.y - v2.e.y, v1.e.z - v2.e.z);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e.x * v2.e.x, v1.e.y * v2.e.y, v1.e.z * v2.e.z);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e.x / v2.e.x, v1.e.y / v2.e.y, v1.e.z / v2.e.z);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v)
{
	return vec3(t * v.e.x, t * v.e.y, t * v.e.z);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
	return vec3(v.e.x / t, v.e.y / t, v.e.z / t);
}

__host__ __device__ inline vec3 operator/(float t, vec3 v)
{
	return vec3(t / v.e.x, t / v.e.y, t / v.e.z);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t)
{
	return vec3(t * v.e.x, t * v.e.y, t * v.e.z);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2)
{
	return v1.e.x * v2.e.x + v1.e.y * v2.e.y + v1.e.z * v2.e.z;
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
	return vec3((v1.e.y * v2.e.z - v1.e.z * v2.e.y),
				(-(v1.e.x * v2.e.z - v1.e.z * v2.e.x)),
				(v1.e.x * v2.e.y - v1.e.y * v2.e.x));
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3& v)
{
	e.x += v.e.x;
	e.y += v.e.y;
	e.z += v.e.z;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v)
{
	e.x *= v.e.x;
	e.y *= v.e.y;
	e.z *= v.e.z;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v)
{
	e.x /= v.e.x;
	e.y /= v.e.y;
	e.z /= v.e.z;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v)
{
	e.x -= v.e.x;
	e.y -= v.e.y;
	e.z -= v.e.z;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
	e.x *= t;
	e.y *= t;
	e.z *= t;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t)
{
	float k = 1.0f / t;

	e.x *= k;
	e.y *= k;
	e.z *= k;
	return *this;
}
// Platform detection
#if 0
// CUDA environment
#define IS_CUDA 1
#else
#define IS_CUDA 0
#endif
#endif

// Fast reciprocal square root function implementation (1/sqrt(x))
__device__ inline float custom_rsqrtf(float x)
{
#if IS_CUDA
	// On CUDA, use the built-in rsqrtf
	return rsqrtf(x);
#else
	// On CPU, implement the fast inverse square root algorithm

	// Method using union for type punning (avoids compiler warnings/errors)
	union
	{
		float f;
		int	  i;
	} conv;

	float x2 = x * 0.5f;
	conv.f	 = x;
	conv.i	 = 0x5f3759df - (conv.i >> 1);
	x		 = conv.f;

	// Newton iterations for refinement
	x = x * (1.5f - x2 * x * x); // First iteration
	// x = x * (1.5f - x2 * x * x);  // Optional second iteration for more precision

	return x;
#endif
}

__device__ inline glm::vec3 unit_vector(const glm::vec3& v)
{
	float invLen = glm::fastInverseSqrt(glm::dot(v, v));
	return {v.x * invLen, v.y * invLen, v.z * invLen};
}

__device__ __forceinline__ glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n)
{
	return v - 2.f * glm::dot(v, n) * n;
}

__device__ __forceinline__ bool refract(const glm::vec3& v, const glm::vec3& n, float niOverNt, glm::vec3& outRefracted)
{
	float dt		   = dot(v, n);
	float discriminant = 1.f - niOverNt * niOverNt * (1.f - dt * dt);
	if (discriminant > 0.f)
	{
		outRefracted = niOverNt * (v - n * dt) - n * sqrtf(discriminant);
		return true;
	}
	return false;
}

__device__ __forceinline__ float Reflectance(float cosine, float refIdx)
{
	// Schlick's approximation
	float r0 = (1.f - refIdx) / (1.f + refIdx);
	r0		 = r0 * r0;
	return r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
}
__device__ __forceinline__ float fastLength(const glm::vec3& v)
{
	// Approximate length using rsqrtf
	float lenSq = dot(v, v);
	return lenSq > 0.f ? lenSq * custom_rsqrtf(lenSq) : 0.f;
}

__host__ __device__ inline glm::vec3 make_unit_vector(glm::vec3 e)
{
	float k = 1.0f / sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
	e.x *= k;
	e.y *= k;
	e.z *= k;
	return {e.x, e.y, e.z};
}


namespace math
{

	__device__ constexpr Float SqrtNewtonRaphson(const Float x, const Float current, const Float prev) noexcept
	{
		return current == prev
				   ? current
				   : SqrtNewtonRaphson(x, (Float)0.5 * (current + x / current), current);
	}

	__device__ constexpr Float Sqrt(const Float x) noexcept
	{
		return x >= 0 && x < CUDART_INF_F
				   ? SqrtNewtonRaphson(x, x, 0)
				   : CUDART_INF_F / CUDART_INF_F; // IEEE-754: NaN
	}

	//__device__ inline bool NearZero(const vec3& v) noexcept
	//{
	//	const auto s = 1e-8f;
	//	return (std::fabs(v.x) < s && std::fabs(v.y) < s && std::fabs(v.z) < s);
	//}
} // namespace math
