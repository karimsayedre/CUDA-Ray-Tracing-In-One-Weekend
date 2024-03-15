#pragma once
#include <vector>



#include <EASTL/vector.h>

#include "Hittable.h"
#include "HittableList.h"



class BVHNode : public Hittable 
{
public:
    BVHNode(const HittableList& list, double time0, double time1)
        : BVHNode(list.m_Objects, 0, list.m_Objects.size(), time0, time1)
    {
    }

    BVHNode(const eastl::vector<Hittable*>& src_objects, size_t start, size_t end, double time0, double time1);

    bool Hit(const Ray& r, const T tMin, T tMax, HitRecord& rec) const
    {
        if (!m_Box.Hit(r, tMin, tMax))
            return false;


        const bool hitLeft = m_Left->Hit(r, tMin, tMax, rec);
        //const auto max = hitLeft ? rec.T : tMax;
        if (hitLeft)
            tMax = rec.T;

        const bool hitRight = m_Right->Hit(r, tMin, tMax, rec);

        return hitLeft || hitRight;
    }

    bool BoundingBox(double time0, double time1, AABB& outputBox) const
    {
        outputBox = m_Box;
        return true;
    }

    Hittable* m_Left;
    Hittable* m_Right;
    AABB m_Box;

};


