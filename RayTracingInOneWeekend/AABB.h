#pragma once
#include <cuda_runtime.h>
// #include <nmmintrin.h>
#include "Interval.h"
#include "Ray.h"

// inline void SwapValues(__m128& a, __m128& b, const __m128 condition) {
//	__m128 temp = b;
//	b = _mm_blendv_ps(b, a, condition);
//	a = _mm_blendv_ps(a, temp, condition);
// }

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
	Interval x, y, z;

	AABB() = default; // The default AABB is empty, since Intervals are empty by default.

	__device__ AABB(const Interval& ix, const Interval& iy, const Interval& iz)
		: x(ix), y(iy), z(iz)
	{
	}

	__device__ AABB(const vec3& a, const vec3& b)
	{
		// Treat the two points a and b as extrema for the bounding box, so we don't require a
		// particular minimum/maximum coordinate order.
		x = Interval(fmin(a[0], b[0]), fmax(a[0], b[0]));
		y = Interval(fmin(a[1], b[1]), fmax(a[1], b[1]));
		z = Interval(fmin(a[2], b[2]), fmax(a[2], b[2]));
	}

	__device__ AABB(const AABB& box0, const AABB& box1)
	{
		x = Interval(box0.x, box1.x);
		y = Interval(box0.y, box1.y);
		z = Interval(box0.z, box1.z);
	}

	__device__ [[nodiscard]] AABB Pad() const
	{
		// Return an AABB that has no side narrower than some delta, padding if necessary.
		Float	 delta = 0.0001f;
		Interval new_x = (x.size() >= delta) ? x : x.Expand(delta);
		Interval new_y = (y.size() >= delta) ? y : y.Expand(delta);
		Interval new_z = (z.size() >= delta) ? z : z.Expand(delta);

		return AABB(new_x, new_y, new_z);
	}

	__device__ [[nodiscard]] const Interval& axis(int n) const
	{
		if (n == 1)
			return y;
		if (n == 2)
			return z;
		return x;
	}

	__device__ [[nodiscard]] bool Hit(const Ray& r, Interval rayT) const
	{
		for (int a = 0; a < 3; a++)
		{
			auto invD = 1.0f / r.Direction()[a];
			auto orig = r.Origin()[a];

			auto t0 = (axis(a).min - orig) * invD;
			auto t1 = (axis(a).max - orig) * invD;

			if (invD < 0)
				std::swap(t0, t1);

			if (t0 > rayT.min)
				rayT.min = t0;
			if (t1 < rayT.max)
				rayT.max = t1;

			if (rayT.max <= rayT.min)
				return false;
		}
		return true;
	}

	__device__ Float DistanceToRayOrigin(const Ray& r) const
	{
		// Compute the vector from ray origin to box center
		vec3 diff = vec3(x.Center(), y.Center(), z.Center()) - r.Origin();

		// Return squared Euclidean distance (cheaper than sqrt)
		return dot(diff, diff);
	}

	__device__ [[nodiscard]] int LongestAxis() const
	{
		// Returns the index of the longest axis of the bounding box.

		if (x.size() > y.size())
			return x.size() > z.size() ? 0 : 2;
		else
			return y.size() > z.size() ? 1 : 2;
	}
};

__device__ inline AABB operator+(const AABB& bbox, const vec3& offset)
{
	return {bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z()};
}

__device__ inline AABB operator+(const vec3& offset, const AABB& bbox)
{
	return bbox + offset;
}

// class AABB
//{
//   public:
//	__device__ vec3 Min() const
//	{
//		return m_Minimum;
//	}
//	__device__ vec3 Max() const
//	{
//		return m_Maximum;
//	}
//
//	__device__ AABB(const vec3& min, const vec3& max)
//		: m_Minimum(min), m_Maximum(max)
//	{
//	}
//
//	AABB() = default;
//
//	__device__ bool Hit(const Ray& r, Float tMin, Float tMax) const noexcept
//	{
//		// for (int a = 0; a < 3; a++)
//		//{
//		//	Float invD = r.InvDirection()[a];
//		//	Float t0   = (m_Minimum[a] - r.Origin()[a]) * invD;
//		//	Float t1   = (m_Maximum[a] - r.Origin()[a]) * invD;
//		//	if (invD < 0.0f)
//		//		std::swap(t0, t1);
//		//	tMin = t0 > tMin ? t0 : tMin;
//		//	tMax = t1 < tMax ? t1 : tMax;
//		//	if (tMax <= tMin)
//		//		return false;
//		// }
//		// return true;
//
//		// const vec3& ray_orig = r.Origin();
//		// const vec3&	  ray_dir  = r.Direction();
//
//		// for (int axis = 0; axis < 3; axis++)
//		//{
//		//	//const Interval& ax	  = axis_Interval(axis);
//		//	const double	adinv = 1.0 / ray_dir[axis];
//
//		//	auto t0 = (ax.min - ray_orig[axis]) * adinv;
//		//	auto t1 = (ax.max - ray_orig[axis]) * adinv;
//
//		//	if (t0 < t1)
//		//	{
//		//		if (t0 > ray_t.min)
//		//			ray_t.min = t0;
//		//		if (t1 < ray_t.max)
//		//			ray_t.max = t1;
//		//	}
//		//	else
//		//	{
//		//		if (t1 > ray_t.min)
//		//			ray_t.min = t1;
//		//		if (t0 < ray_t.max)
//		//			ray_t.max = t0;
//		//	}
//
//		//	if (ray_t.max <= ray_t.min)
//		//		return false;
//		//}
//		// return true;
//
//		const auto lambda = [&]<int Axis>() -> bool
//		{
//			Float invD = r.InvDirection()[Axis];
//			Float t0   = (m_Minimum[Axis] - r.Origin()[Axis]) * invD;
//			Float t1   = (m_Maximum[Axis] - r.Origin()[Axis]) * invD;
//			if (invD < 0.0f)
//			{
//				volatile float temp = t0;
//				t0					= t1;
//				t1					= temp;
//			}
//			tMin = t0 > tMin ? t0 : tMin;
//			tMax = t1 < tMax ? t1 : tMax;
//			return tMax > tMin;
//		};
//
//		if (lambda.operator()<0>())
//			if (lambda.operator()<1>())
//				if (lambda.operator()<2>())
//					return true;
//
//		return false;
//
//		// union { __m128 t0; float t_0[4]; };
//		// union { __m128 t1; float t_1[4]; };
//		// union { __m128 invD; float inv_D[4]; };
//		// invD = _mm_set_ps(1.0f, r.InvDirection().z, r.InvDirection().y, r.InvDirection().x);
//
//		// t0 = _mm_sub_ps(_mm_loadu_ps(&vec4(m_Minimum, 1.0f)[0]), _mm_loadu_ps(&vec4(r.Origin(), 1.0f)[0]));
//		// t1 = _mm_sub_ps(_mm_loadu_ps(&vec4(m_Maximum, 1.0f)[0]), _mm_loadu_ps(&vec4(r.Origin(), 1.0f)[0]));
//
//		// t0 = _mm_mul_ps(t0, invD);
//		// t1 = _mm_mul_ps(t1, invD);
//
//		//__m128 cond = _mm_cmplt_ps(invD, _mm_setzero_ps());
//
//		// SwapValues(t0, t1, cond);
//
//		////tMin = t_0[0] > tMin ? t_0[0] : tMin;
//		////tMin = t_0[1] > tMin ? t_0[1] : tMin;
//		////tMin = t_0[2] > tMin ? t_0[2] : tMin;
//		////
//		////tMax = t_1[0] < tMax ? t_1[0] : tMax;
//		////tMax = t_1[1] < tMax ? t_1[1] : tMax;
//		////tMax = t_1[2] < tMax ? t_1[2] : tMax;
//
//		//__m128 tMinSSE = _mm_max_ps(t0, _mm_set1_ps(tMin));
//		//__m128 tMaxSSE = _mm_min_ps(t1, _mm_set1_ps(tMax));
//
//		//__m128 mask = _mm_cmpgt_ps(tMaxSSE, tMinSSE);
//
//		// return _mm_movemask_ps(mask) == 0x7;
//	}
//
//   private:
//	vec3 m_Minimum;
//	vec3 m_Maximum;
// };
