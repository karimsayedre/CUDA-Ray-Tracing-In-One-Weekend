#pragma once
#include <algorithm>
#include <limits> // Include this header for std::numeric_limits

#include "Vec3.h"

class Interval
{
  public:
	Float min, max;

	__device__ Interval()
		: min(std::numeric_limits<Float>::infinity()), max(-std::numeric_limits<Float>::infinity())
	{
	} // Default Interval is empty

	__device__ Interval(Float _min, Float _max)
		: min(_min), max(_max)
	{
	}

	__device__ Interval(const Interval& a, const Interval& b)
		: min(std::min(a.min, b.min)), max(std::max(a.max, b.max))
	{
	}

	__device__ [[nodiscard]] double size() const
	{
		return max - min;
	}

	__device__ [[nodiscard]] Interval Expand(Float delta) const
	{
		auto padding = delta / 2;
		return {min - padding, max + padding};
	}

	__device__ [[nodiscard]] bool Contains(Float x) const
	{
		return min <= x && x <= max;
	}

	__device__ [[nodiscard]] bool Surrounds(Float x) const
	{
		return min < x && x < max;
	}

	__device__ [[nodiscard]] Float Clamp(Float x) const
	{
		if (x < min)
			return min;
		if (x > max)
			return max;
		return x;
	}

	__device__ [[nodiscard]] float Center() const
	{
		return 0.5f * (min + max);
	}

	static const Interval empty;
	static const Interval universe;
};

inline const Interval Interval::empty	 = Interval(+std::numeric_limits<Float>::infinity(), -std::numeric_limits<Float>::infinity());
inline const Interval Interval::universe = Interval(-std::numeric_limits<Float>::infinity(), +std::numeric_limits<Float>::infinity());

__device__ inline Interval operator+(const Interval& ival, Float displacement)
{
	return Interval(ival.min + displacement, ival.max + displacement);
}

__device__ inline Interval operator+(Float displacement, const Interval& ival)
{
	return ival + displacement;
}