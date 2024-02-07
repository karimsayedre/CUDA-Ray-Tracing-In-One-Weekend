#pragma once
#include <vector>



#include <EASTL/vector.h>

#include "Hittable.h"
#include "HittableList.h"

class BVHNode : public Hittable {
public:
    BVHNode(const HittableList& list, double time0, double time1)
        : BVHNode(list.m_Objects, 0, list.m_Objects.size(), time0, time1)
    {
    }

    BVHNode(const eastl::vector<Hittable*>& src_objects, size_t start, size_t end, double time0, double time1);

    virtual bool Hit(
        const Ray& r, T tMin, T tMax, HitRecord& rec) const override;

    bool BoundingBox(double time0, double time1, AABB& outputBox) const override
    {
        outputBox = m_Box;
        return true;
    }

    Hittable* m_Left;
    Hittable* m_Right;
    AABB m_Box;
};


