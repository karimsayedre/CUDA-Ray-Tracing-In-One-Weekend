#include "Renderer.h"

#include <iostream>
#include <fmt/chrono.h>


sf::Image Renderer::Render(const uint32_t width, const uint32_t height)
{
	// Image
	float aspectRatio = (float)width / (float)height;
	constexpr int samplesPerPixel = 30;
	constexpr float colorMul = 1.0f / (float)samplesPerPixel;
	constexpr int maxDepth = 50;

	// World
	HittableList world = RandomScene();

	// Camera
	Camera camera(glm::vec3(13.0f, 2.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 20.0f, aspectRatio);

	// Render
	sf::Image image;
	image.create(width, height);
	sf::Thread OpenGLThread([width, height, &image]()
		{
			sf::RenderWindow window(sf::VideoMode(width, height), "Ray Tracing In A Weekend!");
			sf::Texture texture;
			texture.create(width, height);
			texture.update(image);
			sf::Sprite sprite(texture);

			while (window.isOpen())
			{
				window.clear(sf::Color::Transparent);
				// check all the window's events that were triggered since the last iteration of the loop
				sf::Event event;
				while (window.pollEvent(event))
				{
					if (event.type == sf::Event::Closed)
					{
						window.close();
						break;
					}
				}

				texture.loadFromImage(image);
				window.draw(sprite);

				window.display();
			}
		});
	OpenGLThread.launch();

	std::vector<std::jthread> threadPool(std::thread::hardware_concurrency());

	const auto start = std::chrono::high_resolution_clock::now();


	for (uint32_t y = 0; y < height; ++y)
	{
		uint32_t thr = 0;
		for (; thr < threadPool.size(); thr++)
		{
			threadPool[thr] = std::jthread([width, height, &image, &camera, &world, this, y = y + thr]
				{
					const float v = ((float)y + RandomFloat()) / static_cast<float>(height - 1u);
					for (uint32_t x = 0; x < width; ++x)
					{
						const float u = ((float)x + RandomFloat()) / static_cast<float>(width - 1u);

						glm::vec3 pixelColor{};
						for (int s = 0; s < samplesPerPixel; ++s)
							pixelColor += RayColor(camera.GetRay(u, 1.0f - v), world, maxDepth);
						pixelColor *= colorMul;

						image.setPixel(x, y, { static_cast<sf::Uint8>(255.f * pixelColor.x), static_cast<sf::Uint8>(255.f * pixelColor.y), static_cast<sf::Uint8>(255.f * pixelColor.z) });
					}
				});
		}
		LOG_CORE_INFO("\rProgress: {}%\r", y / (float)height * 100);
		y += thr - 1;
	}

	const auto end = std::chrono::high_resolution_clock::now();

	LOG_CORE_INFO("Time taken: {:%S}", (end - start));

	OpenGLThread.wait();
	return image;
}
