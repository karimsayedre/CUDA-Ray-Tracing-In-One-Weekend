#pragma once
#include <limits>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

using Float				 = float;
constexpr Float Infinity = std::numeric_limits<Float>::infinity();
constexpr Float Pi		 = 3.1415926535897932385f;
// #define CUDART_INF_F (float)(0x7f800000)

constexpr float CUDART_INF_F = std::numeric_limits<float>::infinity();

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
	__host__ __device__ inline float operator[](int i) const
	{
		switch (i)
		{
			case 0: return e.x;
			case 1: return e.y;
			case 2: return e.z;
		}
		//assert(false);
		return 0;
	}
	__host__ __device__ inline float& operator[](int i)
	{
		switch (i)
		{
			case 0: return e.x;
			case 1: return e.y;
			case 2: return e.z;
		}
		//assert(false);
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

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
	return v / v.length();
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
