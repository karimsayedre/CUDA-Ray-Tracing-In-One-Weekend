#pragma once

#include <mutex>
#include <atomic>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/Image.hpp>
#include "CudaRenderer.h"

extern sf::Image image; // Global to prevent copy when returning from Render function

inline void Render(const uint32_t width, const uint32_t height)
{
	constexpr int	samplesPerPixel = 30;
	constexpr float colorMul		= 1.0f / static_cast<float>(samplesPerPixel);
	constexpr int	maxDepth		= 50;

	std::atomic_bool		  isRunning {true};
	std::mutex				  imageMutex;
	std::atomic<sf::Vector2u> sharedDimensions {{width, height}};
	std::atomic<float>		  frameTime {0.0f};
	std::atomic_bool		  frameReady {false};

	image.resize(sharedDimensions.load());

	std::thread thread([&]
	{
		sf::RenderWindow window(sf::VideoMode(sharedDimensions.load()), "Ray Tracing In One Weekend!");
		sf::Texture		 texture;
		if (!texture.resize(sharedDimensions.load()))
			assert(false && "Failed to create texture");
		sf::Sprite sprite(texture);

		while (window.isOpen())
		{
			frameReady.wait(false);

			while (auto event = window.pollEvent())
			{
				if (event.value().is<sf::Event::Closed>())
				{
					window.close();
					isRunning = false;
				}
				else if (const auto* resized = event->getIf<sf::Event::Resized>())
				{
					const uint32_t newWidth	 = std::max(resized->size.x, 1u);
					const uint32_t newHeight = std::max(resized->size.y, 1u);

					sf::Vector2u newDims = {newWidth, newHeight};

					// Signal the rendering thread first with the new dimensions
					sharedDimensions.store(newDims, std::memory_order_release);

					// Then update UI components
					{
						std::lock_guard lock(imageMutex);
						image.resize(newDims);
						if (!texture.resize(newDims)) // Recreate texture instead of resize
							assert(false && "Failed to create texture");
						texture.setSmooth(true);
						sprite.setTexture(texture, true);
						window.setView(sf::View(sf::FloatRect({0, 0}, {static_cast<float>(newWidth), static_cast<float>(newHeight)})));
					}

					// Maybe add a synchronization point here to ensure renderer picks up the change
					frameReady.store(false, std::memory_order_release);
				}
			}

			{
				std::lock_guard lock(imageMutex);
				if (!texture.loadFromImage(image))
					assert(false && "Failed to load texture");
			}

			const auto [Width, Height] = sharedDimensions.load(std::memory_order_acquire);
			window.setTitle(fmt::format("Ray Tracing! {}x{} | {:.3f}ms", Width, Height, frameTime.load()));
			window.draw(sprite);

			window.display();
			frameReady.store(false, std::memory_order_release);
		}
	});

	CudaRenderer cudaRenderer(sharedDimensions, samplesPerPixel, maxDepth, colorMul);
	sf::Vector2u cachedDims = sharedDimensions.load(std::memory_order_acquire);

	while (isRunning)
	{
		const sf::Vector2u newDims = sharedDimensions.load(std::memory_order_acquire);
		if (newDims.x != cachedDims.x || newDims.y != cachedDims.y)
		{
			cachedDims = newDims;
			cudaRenderer.ResizeImage(newDims);
		}

		frameTime.store(cudaRenderer.Render(cachedDims.x, cachedDims.y).count(), std::memory_order_release);

		{
			std::lock_guard lock(imageMutex);
			if (image.getSize().x == cachedDims.x && image.getSize().y == cachedDims.y)
			{
				CHECK_CUDA_ERRORS(cudaMemcpy2DFromArray((void*)image.getPixelsPtr(), (size_t)cachedDims.x * 4, cudaRenderer.GetImageArray(), 0, 0, (size_t)cachedDims.x * 4, cachedDims.y, cudaMemcpyDeviceToHost));
			}
		}

		frameReady.store(true, std::memory_order_release);
		frameReady.notify_one();
	}

	thread.join();
}