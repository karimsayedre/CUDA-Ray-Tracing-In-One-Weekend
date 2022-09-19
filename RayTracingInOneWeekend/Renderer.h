#pragma once
#define GLM_FORCE_SSE2

#include <SFML/Graphics/Image.hpp>

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

	auto ground_material = std::make_shared<Lambert>(glm::vec3(0.5, 0.5, 0.5));
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

	return world;
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
		auto t = (T)0.5 * (unitDirection.y + (T)1.0);
		return (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
	}

	inline sf::Image Render(const uint32_t width, const uint32_t height)
    {
	    // Image
    	float aspectRatio = (float)width / (float)height;
    	constexpr int samplesPerPixel = 10;
    	constexpr int maxDepth = 50;
		sf::Image image;
		image.create(width, height);
    	// World
    	HittableList world = RandomScene();

		// Camera
		Camera camera(glm::vec3(13.0f, 2.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 20.0f, aspectRatio);


		// Render
		//std::string imageData = fmt::format("P3\n{} {}\n255\n", width, height);
		//LOG_FILE("P3\n{} {}\n255", width, height);

		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				glm::vec3 pixelColor{};
				for (int s = 0; s < samplesPerPixel; ++s)
				{
					auto u = T(i + RandomFloat()) / (width - 1);
					auto v = T(j + RandomFloat()) / (height - 1);
					pixelColor += RayColor(camera.GetRay(u, 1 - v), world, maxDepth);
				}
				image.setPixel(i, j, { static_cast<sf::Uint8>(255.f * pixelColor.x), static_cast<sf::Uint8>(255.f * pixelColor.y), static_cast<sf::Uint8>(255.f * pixelColor.z) });


				//imageData += WriteColor(pixelColor, samplesPerPixel);
				//WriteColor1(pixelColor, samplesPerPixel);
			}
		}
		return image;
	}

};

