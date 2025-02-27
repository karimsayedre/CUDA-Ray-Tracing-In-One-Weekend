#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/vec3.hpp>
#include <glm/ext/quaternion_geometric.hpp>

#include "AABB.h"
#include "Ray.h"

class AABB;
class Sphere;
class HittableList;
class BVHNode;
class Material;

struct HitRecord
{
	vec3			   Location;
	vec3			   Normal;
	Float				   T;
	const Material*		   MaterialPtr;
	bool				   FrontFace;
	__device__ void SetFaceNormal(const Ray& ray, const vec3& outwardNormal);
};

enum class HittableType
{
	eInvalid,
	eBvhNode,
	eHittableList,
	eSphere,
};

class Hittable
{
	HittableType Type;
	union
	{
		BVHNode*	  u_BvhNode;
		HittableList* u_HittableList;
		Sphere*		  u_Sphere;
	};

  public:

	// Constructors for derived types (using forward declarations)
	__device__ Hittable(Sphere* hittable);

	__device__ Hittable(HittableList* hittable);

	__device__ Hittable(BVHNode* hittable);

	// Delete default constructor
	Hittable() = delete;

	// Methods
	__device__ bool Hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const;
	__device__ bool GetBoundingBox(double time0, double time1, AABB& outputBox) const;
	__device__ bool IsLeaf() const
	{
		return Type == HittableType::eSphere;
	}
};
