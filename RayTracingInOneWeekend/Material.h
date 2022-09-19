#pragma once

#include "Hittable.h"
#include "Ray.h"

class Material
{
public:
	virtual bool Scatter(const Ray& ray, const HitRecord& rec, glm::vec3& attenuation, Ray& scattered) const = 0;
};

