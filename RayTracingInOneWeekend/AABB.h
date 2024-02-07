#pragma once

#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nmmintrin.h>
#include "Ray.h"



inline void SwapValues(__m128& a, __m128& b, const __m128 condition) {
	__m128 temp = b;
	b = _mm_blendv_ps(b, a, condition);
	a = _mm_blendv_ps(a, temp, condition);
}


class AABB
{
public:
	[[nodiscard]] glm::vec3 Min() const { return m_Minimum; }
	[[nodiscard]] glm::vec3 Max() const { return m_Maximum; }

	AABB(const glm::vec3& min, const glm::vec3& max)
		: m_Minimum(min), m_Maximum(max)
	{
	}

	AABB() = default;


	[[nodiscard]] bool Hit(const Ray& r, T tMin, T tMax) const noexcept;

private:
	glm::vec3 m_Minimum;
	glm::vec3 m_Maximum;
};

