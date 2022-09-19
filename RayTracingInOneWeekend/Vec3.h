#pragma once
#include <cassert>
#include <cmath>
#include <concepts>
#include <random>
#include <string>
#include "Log.h"
#include <fmt/format.h>

#define GLM_FORCE_SSE2

#define GLM_FORCE_INLINE
#define GLM_MESSAGES GLM_ENABLE
#include <glm/geometric.hpp>
#include <glm/gtx/norm.hpp>


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

inline float RandomFloat(const float min = 0.0f, const float max = 1.0f)
{
	return min + (float)(rand()) / (float)(RAND_MAX / (max - min));
}

inline glm::vec3 RandomVec3(const float min = 0.0f, const float max = 1.0f)
{
	return { RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max) };
}

template <typename OStream, typename T>
inline constexpr OStream& operator<<(OStream& out, const glm::vec3& v) noexcept
{
	return out << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
}


inline glm::vec3 RandomNormalizedVector()
{
	return glm::normalize(RandomVec3());
}




//using Vec3 = vec3;
//using glm::vec3 = vec3;
//using glm::vec3 = vec3;

inline void WriteColor1(const glm::vec3& color, const int samplesPerPixel)
{
	auto scale = 1.0f / (T)samplesPerPixel;
	const glm::vec3 newColor = glm::sqrt(glm::clamp(color * scale, 0.0f, 1.0f));

	LOG_FILE("{} {} {}", static_cast<int>(255.999f * newColor.x), static_cast<int>(255.999f * newColor.y), static_cast<int>(255.999f * newColor.z));
}

inline std::string WriteColor(const glm::vec3& color, const int samplesPerPixel)
{
	auto scale = 1.0f / (T)samplesPerPixel;
	const glm::vec3 newColor = glm::sqrt(glm::clamp(color * scale, 0.0f, 1.0f));
	return fmt::format("{} {} {}\n", static_cast<int>(255.999f * newColor.x), static_cast<int>(255.999f * newColor.y), static_cast<int>(255.999f * newColor.z));
}


