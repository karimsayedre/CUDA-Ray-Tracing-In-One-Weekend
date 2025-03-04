#pragma once

#include <SFML/Graphics/Image.hpp>

#include "Dielectric.h"
#include "HittableList.h"
#include "Lambert.h"
#include "Metal.h"
#include "Sphere.h"
#include "BVH.h"

#include <EASTL/shared_ptr.h>


//inline HittableList RandomScene() {
//	HittableList world;
//
//	auto ground_material = eastl::make_shared<u_Lambert>(vec3(0.5, 0.5, 0.5));
//	world.Add(new Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));
//
//	auto seed = (uint32_t)rand();
//
//	for (int a = -11; a < 11; a++) {
//		for (int b = -11; b < 11; b++) {
//			auto choose_mat = RandomFloat(seed);
//			vec3 center(a + 0.9f * RandomFloat(seed), 0.2f, b + 0.9f * RandomFloat(seed));
//
//			if ((length(center - vec3(4, 0.2f, 0))) > 0.9f) {
//
//				if (choose_mat < 0.8f) {
//					// diffuse
//					auto albedo = RandomVec3(seed) * RandomVec3(seed);
//					auto sphere_material = eastl::make_shared<u_Lambert>(albedo);
//					world.Add(new Sphere(center, 0.2f, sphere_material));
//				}
//				else if (choose_mat < 0.95f) {
//					// metal
//					auto albedo = RandomVec3(seed, 0.5f, 1.0f);
//					auto fuzz = RandomFloat(seed, 0.0f, 0.5f); // 0, 0.5
//					auto sphere_material = eastl::make_shared<u_Metal>(albedo, fuzz);
//					world.Add(new Sphere(center, 0.2f, sphere_material));
//				}
//				else {
//					// glass
//					auto sphere_material = eastl::make_shared<u_Dielectric>(1.5f);
//					world.Add(new Sphere(center, 0.2f, sphere_material));
//				}
//			}
//		}
//	}
//
//	auto material1 = eastl::make_shared<u_Dielectric>(1.5f);
//	world.Add(new Sphere(vec3(0, 1, 0), 1.0f, material1));
//
//	auto material2 = eastl::make_shared<u_Lambert>(vec3(0.4f, 0.2f, 0.1f));
//	world.Add(new Sphere(vec3(-4, 1, 0), 1.0, material2));
//
//	auto material3 = eastl::make_shared<u_Metal>(vec3(0.7f, 0.6f, 0.5f), 0.0f);
//	world.Add(new Sphere(vec3(4, 1, 0), 1.0, material3));
//
//	return { new BVHNode(world, 0.0, 1.0) };
//}


class Renderer
{
public:

	
	//[[nodiscard]] inline static vec3 RayColor(Ray ray, const Hittable& world, const int depth, curandState* local_rand_state)
	//{
	//	vec3 color = vec3{1.0};

	//	vec3 attenuation = vec3{ 1.0f };
	//	for (int i = 0; i < depth; i++) {
	//		HitRecord hitRecord;
	//		if (!world.Hit(ray, 0.1f, Infinity, hitRecord)) {
	//			const vec3 unitDirection = (ray.Direction().make_unit_vector());
	//			auto	   t			 = 0.5f * (unitDirection.y() + 1.0f);
	//			color *= 1.0f - t * vec3(1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
	//			break;
	//		}

	//		Ray scattered;
	//		if (hitRecord.MaterialPtr->Scatter(ray, hitRecord, attenuation, scattered, local_rand_state))
	//		{
	//			color *= attenuation;
	//			ray = scattered;
	//		}
	//		else
	//		{
	//			break;
	//		}
	//		
	//	}

	//	return color * attenuation;
	//}


	sf::Image Render(const uint32_t width, const uint32_t height);
};

