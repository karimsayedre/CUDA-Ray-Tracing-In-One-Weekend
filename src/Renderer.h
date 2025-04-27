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
#include "CudaGLSurface.h"

inline sf::Image Render(const uint32_t initialWidth, const uint32_t initialHeight)
{
	// constant settings
	constexpr uint32_t samplesPerPixel = 30;
	constexpr float	   colorMul		   = 1.0f / static_cast<float>(samplesPerPixel);
	constexpr uint32_t maxDepth		   = 50;

	// Shared state
	std::atomic<bool>		  isRunning {true};
	std::atomic<sf::Vector2u> sharedDimensions {{initialWidth, initialHeight}};
	float					  timeTaken {0.0f}; // no need to be atomic, performance cost for no reason.

	// Create SFML window in main thread, then deactivate context
	sf::RenderWindow window(sf::VideoMode(sharedDimensions), "Ray Tracing In One Weekend!", sf::Style::Default, sf::State::Windowed, sf::ContextSettings {.depthBits = 0, .stencilBits = 0, .antiAliasingLevel = 0, .majorVersion = 4, .minorVersion = 3, .attributeFlags = sf::ContextSettings::Default, .sRgbCapable = false});
	CHECK_BOOL(window.setActive(false));
	sf::Texture texture;

	// Thread: OpenGL context + CUDA rendering
	std::jthread renderThread([&]
	{
		// Activate GL context on this thread
		CHECK_BOOL(window.setActive(true));

		CHECK_BOOL(texture.resize(sharedDimensions));
		//texture.setSmooth(true);
		CudaGLSurface cudaSurface(texture.getNativeHandle());

		CudaRenderer cudaRenderer(sharedDimensions, samplesPerPixel, maxDepth, colorMul);
		sf::Sprite	 sprite(texture);

		while (isRunning.load(std::memory_order_relaxed))
		{
			{
				cudaSurfaceObject_t surface = cudaSurface.BeginFrame();
				timeTaken					= cudaRenderer.Render(sharedDimensions, surface).count();
				cudaSurface.EndFrame(surface);

				window.draw(sprite);
				window.display();
			}

			// Resize texture if dimensions have changed
			if (const auto dims = sharedDimensions.load(std::memory_order_acquire); dims.x != texture.getSize().x || dims.y != texture.getSize().y)
			{
				cudaSurface.UnregisterSurface();
				CHECK_BOOL(texture.resize(dims));
				sprite.setTexture(texture, true);
				//texture.setSmooth(true);
				window.setView(sf::View(sf::FloatRect({0.f, 0.f}, (sf::Vector2f)dims)));
				cudaSurface.Resize(texture.getNativeHandle());
				cudaRenderer.ResizeImage(dims, 0);
			}
		}
	});

	while (true)
	{
		if (auto event = window.waitEvent(sf::milliseconds(16)))
		{
			if (event.value().is<sf::Event::Closed>())
			{
				window.close();
				isRunning = false;
				break;
			}
			else if (const auto* resized = event->getIf<sf::Event::Resized>())
			{
				//  Signal the rendering thread first with the new dimensions
				sharedDimensions.store({std::max(resized->size.x, 1u), std::max(resized->size.y, 1u)}, std::memory_order_release);
			}
		}
		auto dims = sharedDimensions.load(std::memory_order_relaxed);
		window.setTitle(fmt::format("Ray Tracing! {}x{} | GPU Time: {:.3f}ms", dims.x, dims.y, timeTaken));
	}

	return texture.copyToImage(); // GPU to CPU memory
}
