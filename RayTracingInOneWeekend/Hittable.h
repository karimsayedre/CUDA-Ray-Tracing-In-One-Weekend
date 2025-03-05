#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/vec3.hpp>
#include <glm/ext/quaternion_geometric.hpp>

#include "AABB.h"
#include "Ray.h"

class Material;

struct HitRecord
{
	glm::vec3       Location;
	Float           T;
	glm::vec3       Normal;
	uint16_t MaterialIndex;
	//bool			FrontFace;
	//__device__ void SetFaceNormal(const Ray& ray, const vec3& outwardNormal)
	//{
		//FrontFace = dot(ray.Direction(), outwardNormal) < 0.0f;
		//Normal	  = FrontFace ? outwardNormal : -outwardNormal;
	//}
};

// enum class HittableType
//{
//	eInvalid,
//	eBvhNode,
//	eHittableList,
//	eSphere,
// };

class Hittable
{
	// HittableType Type;
	// union
	//{
	//	BVHNode*	  u_BvhNode;
	//	HittableList* u_HittableList;
	//	Sphere*		  u_Sphere;
	// };

  public:
	//__device__ Hittable(Sphere* hittable)
	//	: Type(HittableType::eSphere), u_Sphere(hittable)
	//{
	//}

	//__device__ Hittable(HittableList* hittable)
	//	: Type(HittableType::eHittableList), u_HittableList(hittable)
	//{
	//}

	//__device__ Hittable(BVHNode* hittable)
	//	: Type(HittableType::eBvhNode), u_BvhNode(hittable)
	//{
	//}

	// Delete default constructor
	//Hittable() = delete;

	// Methods
	__device__ [[nodiscard]] __noinline__ virtual bool Hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const = 0;
	__device__ [[nodiscard]] virtual const AABB& GetBoundingBox(double time0, double time1) const					   = 0;
	__device__ [[nodiscard]] virtual bool IsLeaf() const													   = 0;

	virtual ~Hittable() = default;
};
