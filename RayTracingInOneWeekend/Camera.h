#pragma once
#include <glm/vec3.hpp>

#include "Ray.h"

class Camera
{
public:
	Camera(const glm::vec3& lookFrom, const glm::vec3& lookAt, const glm::vec3& up, const T vFov, const T aspectRatio)
	{
        const auto theta = glm::radians(vFov);
        const auto h = std::tan(theta / 2.0f);
        const auto viewportHeight = 2.0f * h;
        const auto viewportWidth = aspectRatio * viewportHeight;

        const auto w = normalize(lookFrom - lookAt);
        const auto u = normalize(cross(up, w));
        const auto v = cross(w, u);


        m_Origin = lookFrom;
        m_Horizontal = u * viewportWidth;
        m_Vertical = v * viewportHeight;
        m_LowerLeftCorner = m_Origin - m_Horizontal / 2.0f - m_Vertical / 2.0f - w;
	}

    Ray GetRay(const T u, const T v)
	{
        return { m_Origin, m_LowerLeftCorner + u * m_Horizontal + v * m_Vertical - m_Origin };
	}

private:
    glm::vec3 m_Origin;
    glm::vec3 m_LowerLeftCorner;
    glm::vec3 m_Horizontal;
    glm::vec3 m_Vertical;
};

