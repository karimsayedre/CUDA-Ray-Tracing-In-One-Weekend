#pragma once
#include <glm/vec3.hpp>

#include "Ray.h"


class AABB
{
public:
	glm::vec3 Min() const { return m_Minimum; }
	glm::vec3 Max() const { return m_Maximum; }

	AABB(const glm::vec3& min, const glm::vec3& max)
		: m_Minimum(min), m_Maximum(max)
	{
	}

	AABB() = default;

	bool Hit(const Ray& r, T tMin, T tMax) const
	{
		for (int a = 0; a < 3; a++) {
			auto invD = 1.0f / r.Direction()[a];
			auto t0 = (m_Minimum[a] - r.Origin()[a]) * invD;
			auto t1 = (m_Maximum[a] - r.Origin()[a]) * invD;
			if (invD < 0.0f)
				std::swap(t0, t1);
			tMin = t0 > tMin ? t0 : tMin;
			tMax = t1 < tMax ? t1 : tMax;
			if (tMax <= tMin)
				return false;
		}
		return true;
	}

private:
	glm::vec3 m_Minimum;
	glm::vec3 m_Maximum;
};

