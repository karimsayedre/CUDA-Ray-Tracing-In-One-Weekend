#include "BVH.h"

inline bool box_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b, int axis) {
    AABB box_a;
    AABB box_b;

    if (!a->BoundingBox(0, 0, box_a) || !b->BoundingBox(0, 0, box_b))
        LOG_CORE_ERROR("No bounding box in bvh_node constructor.");

    return box_a.Min()[axis] < box_b.Min()[axis];
}


bool box_x_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return box_compare(a, b, 0);
}

bool box_y_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return box_compare(a, b, 1);
}

bool box_z_compare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
    return box_compare(a, b, 2);
}

BVHNode::BVHNode(const std::vector<std::shared_ptr<Hittable>>& src_objects, size_t start, size_t end, double time0,
	double time1)
{
    auto objects = src_objects; // Create a modifiable array of the source scene objects

    int axis = RandomInt(0, 2);
    auto comparator = (axis == 0) ? box_x_compare
        : (axis == 1) ? box_y_compare
        : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        m_Left = m_Right = objects[start];
    }
    else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            m_Left = objects[start];
            m_Right = objects[start + 1];
        }
        else {
            m_Left = objects[start + 1];
            m_Right = objects[start];
        }
    }
    else {
        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        auto mid = start + object_span / 2;
        m_Left = make_shared<BVHNode>(objects, start, mid, time0, time1);
        m_Right = make_shared<BVHNode>(objects, mid, end, time0, time1);
    }

    AABB box_left, box_right;

    if (!m_Left->BoundingBox(time0, time1, box_left) || !m_Right->BoundingBox(time0, time1, box_right))
        LOG_CORE_ERROR("No bounding box in bvh_node constructor.");

    m_Box = SurroundingBox(box_left, box_right);
}

bool BVHNode::Hit(const Ray& r, T tMin, T tMax, HitRecord& rec) const
{
    if (!m_Box.Hit(r, tMin, tMax))
        return false;

    const bool hitLeft = m_Left->Hit(r, tMin, tMax, rec);
    const bool hitRight = m_Right->Hit(r, tMin, hitLeft ? rec.T : tMax, rec);

    return hitLeft || hitRight;
}
