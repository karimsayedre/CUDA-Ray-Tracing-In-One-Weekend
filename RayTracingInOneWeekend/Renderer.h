#pragma once

#include <mutex>
#include <atomic>
#include <SFML/System/Thread.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics/Image.hpp>
#include "CudaRenderer.cuh"

#include <atomic>
#include <mutex>
#include <SFML/Graphics.hpp>
#include <fmt/core.h>

inline static sf::Image image; // Global to prevent copy when returning from Render function

inline void Render(const uint32_t width, const uint32_t height)
{
	constexpr int	samplesPerPixel = 30;
	constexpr float colorMul		= 1.0f / static_cast<float>(samplesPerPixel);
	constexpr int	maxDepth		= 50;

	std::atomic_bool		isRunning {true};
	std::atomic<Dimensions> sharedDimensions {{width, height}};
	std::atomic_bool		imageResized {false};
	std::mutex				imageMutex;
	std::atomic<float>		frameTime {0.0f};
	std::atomic_bool		frameReady {false};

	image.create(sharedDimensions.load().Width, sharedDimensions.load().Height);

	sf::Thread thread([&]
	{
		sf::RenderWindow window(sf::VideoMode(sharedDimensions.load().Width, sharedDimensions.load().Height), "Ray Tracing In A Weekend!");
		sf::Texture		 texture;
		texture.create(sharedDimensions.load().Width, sharedDimensions.load().Height);
		sf::Sprite sprite(texture);

		while (window.isOpen())
		{
			frameReady.wait(false);

			sf::Event event;
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
				{
					window.close();
					isRunning = false;
				}
				else if (event.type == sf::Event::Resized)
				{
					const auto newWidth	 = std::max(event.size.width, 1u);
					const auto newHeight = std::max(event.size.height, 1u);

					{
						std::lock_guard lock(imageMutex);
						image.create(newWidth, newHeight);
						texture.create(newWidth, newHeight);
						sprite.setTexture(texture, true);
					}

					window.setView(sf::View(sf::FloatRect(0, 0, static_cast<float>(newWidth), static_cast<float>(newHeight))));

					sharedDimensions.store({newWidth, newHeight}, std::memory_order_release);
					imageResized.store(true, std::memory_order_release);
				}
			}

			{
				std::lock_guard lock(imageMutex);
				texture.loadFromImage(image);
			}

			const auto dims = sharedDimensions.load(std::memory_order_acquire);
			window.setTitle(fmt::format("Ray Tracing! {}x{} | {:.3f}ms", dims.Width, dims.Height, frameTime.load()));
			window.draw(sprite);

			window.display();
			frameReady.store(false, std::memory_order_release);
		}
	});

	thread.launch();

	CudaRenderer cudaRenderer(sharedDimensions, samplesPerPixel, maxDepth, colorMul);
	Dimensions	 cachedDims = sharedDimensions.load(std::memory_order_acquire);

	while (isRunning)
	{
		if (imageResized.exchange(false, std::memory_order_acq_rel))
		{
			const auto newDims = sharedDimensions.load(std::memory_order_acquire);
			if (newDims.Width != cachedDims.Width || newDims.Height != cachedDims.Height)
			{
				cachedDims = newDims;
				cudaRenderer.ResizeImage(newDims.Width, newDims.Height);
			}
		}

		frameTime.store(cudaRenderer.Render(cachedDims.Width, cachedDims.Height).count(), std::memory_order_release);

		{
			std::lock_guard lock(imageMutex);
			if (image.getSize().x == cachedDims.Width && image.getSize().y == cachedDims.Height)
			{
				CHECK_CUDA_ERRORS(cudaMemcpy2DFromArray((void*)image.getPixelsPtr(), cachedDims.Width * 4, cudaRenderer.GetImageArray(), 0, 0, cachedDims.Width * 4, cachedDims.Height, cudaMemcpyDeviceToHost));
			}
		}

		frameReady.store(true, std::memory_order_release);
		frameReady.notify_one();
	}

	thread.wait();
}