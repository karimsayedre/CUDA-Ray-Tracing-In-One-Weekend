#pragma once
#include <utility>
#include <variant>
#include <EASTL/shared_ptr.h>
#include <EASTL/vector.h>

#include "Hittable.h"
#include "AABB.h"

class HittableList : public Hittable
{
public:
    //constexpr HittableType s_HittableType = HittableType::HittableList;

    HittableList()
	    : Hittable(this)
	{}
    HittableList(Hittable* object)
        : Hittable(this)
    {
	    Add(object);
    }

    void Clear() { m_Objects.clear(); }
    void Add(Hittable* object) { m_Objects.emplace_back(object); }

    bool Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const;

    bool BoundingBox(double time0, double time1, AABB& outputBox) const;



    eastl::vector<Hittable*> m_Objects;

};

inline AABB SurroundingBox(const AABB& box0, const AABB& box1)
{
    glm::vec3 small(fmin(box0.Min().x, box1.Min().x),
        fmin(box0.Min().y, box1.Min().y),
        fmin(box0.Min().z, box1.Min().z));

    glm::vec3 big(fmax(box0.Max().x, box1.Max().x),
        fmax(box0.Max().y, box1.Max().y),
        fmax(box0.Max().z, box1.Max().z));

    return { small, big };
}

inline bool HittableList::BoundingBox(double time0, double time1, AABB& outputBox) const
{
    if (m_Objects.empty()) return false;

    AABB tempBox;
    bool firstBox = true;

    for (const auto& object : m_Objects) {
        if (!object->BoundingBox(time0, time1, tempBox)) return false;
        outputBox = firstBox ? tempBox : SurroundingBox(outputBox, tempBox);
        firstBox = false;
    }

    return true;
}

inline bool HittableList::Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const
{
    HitRecord tempRecord;
    bool hitAnything = false;
    T closestSoFar = tMax;

    for(const auto& object : m_Objects)
    {
        if(object->Hit(ray, tMin, closestSoFar, tempRecord))
        {
            hitAnything = true;
            closestSoFar = tempRecord.T;
            record = tempRecord;
        }
    }

    return hitAnything;
}

