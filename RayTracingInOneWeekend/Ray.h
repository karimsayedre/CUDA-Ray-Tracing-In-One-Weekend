#pragma once
#include "Vec3.h"

class Ray
{
public:
	[[nodiscard]] constexpr Ray() noexcept : m_Origin(), m_Direction(), m_InvDirection() {}
	[[nodiscard]] Ray(const glm::vec3& origin, const glm::vec3& direction) noexcept
		: m_Origin(origin), m_Direction(direction), m_InvDirection(1.0f / direction)
	{
	}

	[[nodiscard]] constexpr glm::vec3 Origin() const noexcept { return m_Origin; }
	[[nodiscard]] constexpr glm::vec3 Direction() const noexcept { return m_Direction; }
	[[nodiscard]] constexpr glm::vec3 InvDirection() const noexcept { return m_InvDirection; }

	[[nodiscard]] glm::vec3 At(const T t) const noexcept
	{
		return m_Origin + t * m_Direction;
	}

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
	glm::vec3 m_InvDirection;
};

