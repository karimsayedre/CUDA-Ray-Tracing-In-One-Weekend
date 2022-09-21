#pragma once
#define GLM_FORCE_SSE2

#include <SFML/Graphics/Image.hpp>
#include <thread>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/System/Thread.hpp>
#include <SFML/Window/Event.hpp>

#include "BVH.h"
#include "Camera.h"
#include "Dielectric.h"
#include "Hittable.h"
#include "HittableList.h"
#include "Lambert.h"
#include "Material.h"
#include "Metal.h"
#include "Random.h"
#include "Ray.h"
#include "Sphere.h"
#include "Vec3.h"

inline HittableList RandomScene() {
	HittableList world;

	auto ground_material = std::make_shared<Metal>(glm::vec3(1.0f, 0.0f, 0.0f), 0.3f);
	world.Add(std::make_shared<Sphere>(glm::vec3(0, -1000, 0), 1000, ground_material));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = RandomFloat();
			glm::vec3 center(a + 0.9f * RandomFloat(), 0.2f, b + 0.9f * RandomFloat());

			if ((glm::length(center - glm::vec3(4, 0.2f, 0))) > 0.9f) {
				std::shared_ptr<Material> sphere_material;

				if (choose_mat < 0.8f) {
					// diffuse
					auto albedo = RandomVec3() * RandomVec3();
					sphere_material = std::make_shared<Lambert>(albedo);
					world.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
				}
				else if (choose_mat < 0.95f) {
					// metal
					auto albedo = RandomVec3(0.5f, 1.0f);
					auto fuzz = RandomFloat(0.0f, 0.5f); // 0, 0.5
					sphere_material = std::make_shared<Metal>(albedo, fuzz);
					world.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
				}
				else {
					// glass
					sphere_material = std::make_shared<Dielectric>(1.5);
					world.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
				}
			}
		}
	}

	auto material1 = std::make_shared<Dielectric>(1.5f);
	world.Add(std::make_shared<Sphere>(glm::vec3(0, 1, 0), 1.0f, material1));

	auto material2 = std::make_shared<Lambert>(glm::vec3(0.4f, 0.2f, 0.1f));
	world.Add(std::make_shared<Sphere>(glm::vec3(-4, 1, 0), 1.0, material2));

	auto material3 = std::make_shared<Metal>(glm::vec3(0.7f, 0.6f, 0.5f), 0.0f);
	world.Add(std::make_shared<Sphere>(glm::vec3(4, 1, 0), 1.0, material3));

	return { std::make_shared<BVHNode>(world, 0.0, 1.0) };
}


class Renderer
{
public:


	[[nodiscard]] inline const glm::vec3 RayColor(const Ray& ray, const Hittable& world, const int depth)
	{
		HitRecord hitRecord;
		if (depth <= 0) return glm::vec3{};

		if (world.Hit(ray, 0.001f, Infinity, hitRecord))
		{
			Ray scattered;
			glm::vec3 attenuation;
			if (hitRecord.MaterialPtr->Scatter(ray, hitRecord, attenuation, scattered))
				return attenuation * RayColor(scattered, world, depth - 1);
			return {};
		}
		const glm::vec3 unitDirection = glm::normalize(ray.Direction());
		auto t = 0.5f * (unitDirection.y + 1.0f);
		return (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
	}

	void Render(const uint32_t width, const uint32_t height);
};

