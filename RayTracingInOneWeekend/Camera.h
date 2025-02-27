#pragma once
#include <vector_functions.h>
#include "CudaMath.cuh"
#include "Ray.h"
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>


class Camera
{
public:
	Camera(const vec3& lookFrom, const vec3& lookAt, const vec3& up, const Float vFov, const Float aspectRatio)
	{
        const auto theta          = glm::radians(vFov);
        const auto h              = std::tan(theta / 2.0f);
        const auto viewportHeight = 2.0f * h;
        const auto viewportWidth  = aspectRatio * viewportHeight;

        const auto w = (lookFrom - lookAt).make_unit_vector();
        const auto u = (cross(up, w)).make_unit_vector();
        const auto v = cross(w, u);


        m_Origin = lookFrom;
        m_Horizontal = u * viewportWidth;
        m_Vertical = v * viewportHeight;
        m_LowerLeftCorner = m_Origin - m_Horizontal / 2.0f - m_Vertical / 2.0f - w;
	}

    Ray GetRay(const Float u, const Float v)
	{
        return { m_Origin, m_LowerLeftCorner + u * m_Horizontal + v * m_Vertical - m_Origin };
	}



private:
    vec3 m_Origin;
    vec3 m_LowerLeftCorner;
    vec3 m_Horizontal;
    vec3 m_Vertical;
};

