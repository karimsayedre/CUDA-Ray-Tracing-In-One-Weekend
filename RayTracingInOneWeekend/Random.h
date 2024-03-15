#pragma once
#include "Vec3.h"



inline uint32_t pcg_hash(uint32_t input)
{
	uint32_t state = input * 747796405u + 2891336453u;
	uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

inline float RandomFloat(uint32_t& seed, const float min = 0.0f, const float max = 1.0f)
{
	seed = pcg_hash(seed);
	return min + (float)(seed) / (float)(UINT_MAX / (max - min));
}

inline int RandomInt(uint32_t& seed, const int min, const int max)
{
	return static_cast<int>(RandomFloat(seed, (float)min, (float)max + 1.0f));
}


inline glm::vec3 RandomVec3(uint32_t& seed, const float min = 0.0f, const float max = 1.0f)
{
	return { RandomFloat(seed, min, max), RandomFloat(seed, min, max), RandomFloat(seed, min, max) };
}

template <typename OStream, typename T>
inline constexpr OStream& operator<<(OStream& out, const glm::vec3& v) noexcept
{
	return out << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
}


inline glm::vec3 RandomNormalizedVector(uint32_t& seed)
{
	return glm::normalize(RandomVec3(seed));
}