#pragma once
#include <memory>
#include <vector>

#include "Hittable.h"
#include "HittableList.h"

class BVHNode : public Hittable {
public:
    BVHNode() = default;

    BVHNode(const HittableList& list, double time0, double time1)
        : BVHNode(list.m_Objects, 0, list.m_Objects.size(), time0, time1)
    {}

    BVHNode(const std::vector<std::shared_ptr<Hittable>>& src_objects, size_t start, size_t end, double time0, double time1);

    virtual bool Hit(
        const Ray& r, T tMin, T tMax, HitRecord& rec) const override;

    bool BoundingBox(double time0, double time1, AABB& outputBox) const override
    {
        outputBox = m_Box;
        return true;
    }

    std::shared_ptr<Hittable> m_Left;
    std::shared_ptr<Hittable> m_Right;
    AABB m_Box;
};


