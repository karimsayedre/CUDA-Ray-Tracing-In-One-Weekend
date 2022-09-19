#pragma once
#include "Hittable.h"
class HittableList : public Hittable
{
public:
    HittableList() {}
    HittableList(std::shared_ptr<Hittable> object) { Add(object); }

    inline constexpr void Clear() { m_Objects.clear(); }
    inline void Add(std::shared_ptr<Hittable> object) { m_Objects.emplace_back(object); }

    inline constexpr virtual bool Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const override;

public:
    std::vector<std::shared_ptr<Hittable>> m_Objects;

};

constexpr bool HittableList::Hit(const Ray& ray, const T tMin, const T tMax, HitRecord& record) const
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

