#pragma once

class Interval
{
  public:
	float Min, Max;

	__device__ Interval()
		: Min(std::numeric_limits<float>::infinity()), Max(-std::numeric_limits<float>::infinity())
	{
	} 

	__device__ Interval(float _min, float _max)
		: Min(_min), Max(_max)
	{
	}

	__device__ Interval(const Interval& a, const Interval& b)
		: Min(glm::min(a.Min, b.Min)), Max(glm::max(a.Max, b.Max))
	{
	}

	__device__ [[nodiscard]] float Size() const
	{
		return Max - Min;
	}

	__device__ [[nodiscard]] bool Contains(const float x) const
	{
		return Min <= x && x <= Max;
	}

	__device__ [[nodiscard]] bool Surrounds(const float x) const
	{
		return Min < x && x < Max;
	}

	__device__ [[nodiscard]] float Clamp(const float x) const
	{
		if (x < Min)
			return Min;
		if (x > Max)
			return Max;
		return x;
	}

	__device__ [[nodiscard]] float Center() const
	{
		return 0.5f * (Min + Max);
	}

	static const Interval s_Empty;
	static const Interval s_Universe;
};

inline const Interval Interval::s_Empty	   = Interval(+std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
inline const Interval Interval::s_Universe = Interval(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

__device__ inline Interval operator+(const Interval& interval, float displacement)
{
	return {interval.Min + displacement, interval.Max + displacement};
}

__device__ inline Interval operator+(const float displacement, const Interval& interval)
{
	return interval + displacement;
}