#include "Sphere.h"


bool Sphere::Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const
{
    glm::vec3 oc = ray.Origin() - m_Center;
    auto a = glm::length2(ray.Direction());
    auto half_b = glm::dot(oc, ray.Direction());
    auto c = glm::length2(oc) - m_Radius * m_Radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < tMin || tMax < root) {
        root = (-half_b + sqrtd) / a;
        if (root < tMin || tMax < root)
            return false;
    }

    record.T = root;
    record.Location = ray.At(record.T);
    record.SetFaceNormal(ray, (record.Location - m_Center) / m_Radius);
    record.MaterialPtr = m_MaterialPtr;
    return true;
}

bool Sphere::BoundingBox(double time0, double time1, AABB& outputBox) const
{
    outputBox = AABB(
        m_Center - glm::vec3(m_Radius, m_Radius, m_Radius),
        m_Center + glm::vec3(m_Radius, m_Radius, m_Radius));
    return true;
}
