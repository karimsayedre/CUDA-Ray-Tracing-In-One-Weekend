
#include "pch.cuh"

#include "Renderer.h"
#include <iostream>
#include <fmt/chrono.h>
#include <SFML/Graphics.hpp>
#include "Camera.h"
#include "CudaRenderer.cuh"
#include "Log.h"

sf::Image Renderer::Render(const uint32_t width, const uint32_t height)
{
	// Image
	const float		aspectRatio		= (float)width / (float)height;
	constexpr int	samplesPerPixel = 50;
	constexpr float colorMul		= 1.0f / (float)samplesPerPixel;
	constexpr int	maxDepth		= 50;

	// World
	// HittableList world = RandomScene();

	// Camera
	//Camera camera(vec3(13.0f, 2.0f, 3.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), 20.0f, aspectRatio);

	std::atomic_bool isRunning = true;

	std::mutex				frameMutex;
	std::condition_variable frameCV;
	std::atomic_bool		frameReady {false};
	// Render
	sf::Image image;
	image.create(width, height);
	sf::Thread OpenGLThread([&]()
	{
		sf::RenderWindow window(sf::VideoMode(width, height), "Ray Tracing In A Weekend!");
		sf::Texture		 texture;
		texture.create(width, height);
		texture.update(image);
		sf::Sprite sprite(texture);

		while (window.isOpen())
		{
			// Wait until the main thread signals a new frame is ready.
			{
				std::unique_lock<std::mutex> lock(frameMutex);
				frameCV.wait(lock, [&frameReady]()
				{
					return frameReady.load();
				});
				frameReady = false; // Reset for next frame.
			}

			// window.clear(sf::Color::Transparent);
			//  check all the window's events that were triggered since the last iteration of the loop
			sf::Event event;
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
				{
					window.close();
					isRunning = false;
					// exit(0);
					break;
				}
			}

			texture.loadFromImage(image);
			window.draw(sprite);

			window.display();
		}
	});
	OpenGLThread.launch();

	// std::vector<std::jthread> threadPool(std::thread::hardware_concurrency());

	// while (false)
	//{
	//	static std::atomic<uint32_t> frame = 0;
	//	const auto					 start = std::chrono::high_resolution_clock::now();

	//	for (uint32_t y = 0; y < height;)
	//	{
	//		uint32_t thr = 0;
	//		for (; thr < threadPool.size(); thr++)
	//		{
	//			threadPool[thr] = std::jthread([width, height, &image, &camera, &world, this, y = y + thr, frame = frame.load()]
	//			{
	//				for (uint32_t x = 0; x < width; ++x)
	//				{
	//					uint32_t	seed = y * width + x * frame * samplesPerPixel;
	//					const float v	 = ((float)y + RandomFloat(seed)) / static_cast<float>(height - 1u);
	//					const float u	 = ((float)x + RandomFloat(seed)) / static_cast<float>(width - 1u);

	//					vec3 pixelColor {};
	//					for (int s = 0; s < samplesPerPixel; ++s)
	//						pixelColor += RayColor(camera.GetRay(u, 1.0f - v), world, maxDepth);
	//					pixelColor *= colorMul;

	//					if (frame < 1)
	//					{
	//						image.setPixel(x, y, sf::Color {static_cast<sf::Uint8>(255.f * pixelColor.x), static_cast<sf::Uint8>(255.f * pixelColor.y), static_cast<sf::Uint8>(255.f * pixelColor.z)});
	//						break;
	//					}

	//					sf::Color oldColor = image.getPixel(x, y);

	//					vec3 color = (vec3(oldColor.r / 255.0f, oldColor.g / 255.0f, oldColor.b / 255.0f) + pixelColor) * (0.5f);

	//					image.setPixel(x, y, sf::Color {sf::Uint8(color.r * 255), sf::Uint8(color.g * 255), sf::Uint8(color.b * 255)});
	//				}
	//			});
	//		}
	//		LOG_CORE_INFO("\rProgress: {}%\r", y / (float)height * 100);
	//		y += thr;
	//	}

	//	const auto end = std::chrono::high_resolution_clock::now();

	//	LOG_CORE_INFO("Time taken: {:%S}", (end - start));
	//	++frame;
	//}

	CudaRenderer cudaRenderer(width, height, samplesPerPixel, maxDepth, colorMul);
	cudaRenderer.Init();
	while (isRunning)
	{
		cudaRenderer.Render();

		const std::vector pixels = cudaRenderer.CopyImage();

		for (uint32_t y = 0; y < height; ++y)
		{
			for (uint32_t x = 0; x < width; ++x)
			{
				uint32_t  index = y * width + x;
				sf::Color color;
				color.r = static_cast<uint8_t>(pixels[index * 3] * 255.0f);
				color.g = static_cast<uint8_t>(pixels[index * 3 + 1] * 255.0f);
				color.b = static_cast<uint8_t>(pixels[index * 3 + 2] * 255.0f);
				color.a = 255;
				image.setPixel(x, y, color);
			}
		}
		// Notify the OpenGL thread a new frame is ready:
		{
			std::lock_guard<std::mutex> lock(frameMutex);
			frameReady = true;
		}
		frameCV.notify_one();
		//isRunning = false;
	}

	OpenGLThread.wait();

	return image;
}
