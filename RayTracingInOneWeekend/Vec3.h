#pragma once
#include <cmath>
#include "Log.h"

#include <glm/geometric.hpp>
#include <glm/gtx/norm.hpp>
#include "Random.h"


using T = float;
constexpr T Infinity = std::numeric_limits<T>::infinity();
constexpr T Pi = 3.1415926535897932385f;



namespace math {

	constexpr T SqrtNewtonRaphson(const T x, const T current, const T prev) noexcept
	{
		return current == prev
			? current
			: SqrtNewtonRaphson(x, (T)0.5 * (current + x / current), current);
	}

	constexpr T Sqrt(const T x) noexcept
	{
		return x >= 0 && x < std::numeric_limits<T>::infinity()
			? SqrtNewtonRaphson(x, x, 0)
			: std::numeric_limits<T>::quiet_NaN();
	}

	inline bool NearZero(const glm::vec3& v) noexcept
	{
		const auto s = 1e-8f;
		return (std::fabs(v.x) < s && std::fabs(v.y) < s && std::fabs(v.z) < s);
	}
}

