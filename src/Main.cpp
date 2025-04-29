#include "pch.h"
#include "CudaGLSurface.h"
#include "Renderer.h"
#include "Log.h"
#include "Raytracing.h"
#include "SFML/Window/Event.hpp"
#include "SFML/Graphics.hpp"

static sf::Image Render(const uint32_t initialWidth, const uint32_t initialHeight)
{
	// constant settings
	constexpr uint32_t samplesPerPixel = 30;
	constexpr float	   colorMul		   = 1.0f / static_cast<float>(samplesPerPixel);
	constexpr uint32_t maxDepth		   = 50;

	// Shared state
	std::atomic<bool>		   isRunning { true };
	std::atomic<sf::Vector2u>  sharedDimensions { { initialWidth, initialHeight } };
	float					   timeTaken { 0.0f }; // no need to be atomic, performance cost for no reason.
	std::atomic<ExecutionMode> executionMode { ExecutionMode::GPU };
	bool					   moveCamera = false;

	// Create SFML window in main thread, then deactivate context
	sf::RenderWindow window(sf::VideoMode(sharedDimensions), "Ray Tracing In One Weekend!", sf::Style::Default, sf::State::Windowed, sf::ContextSettings { .depthBits = 0, .stencilBits = 0, .antiAliasingLevel = 0, .majorVersion = 4, .minorVersion = 3, .attributeFlags = sf::ContextSettings::Default, .sRgbCapable = false });
	CHECK_BOOL(window.setActive(false));
	sf::Texture texture;
	sf::Image	image(sharedDimensions);

	// Thread: OpenGL context + CUDA rendering
	std::jthread renderThread([&]
	{
		// Activate GL context on this thread
		CHECK_BOOL(window.setActive(true));

		CHECK_BOOL(texture.resize(sharedDimensions));
		// texture.setSmooth(true);
		CudaGLSurface cudaSurface(texture.getNativeHandle());

		Renderer<ExecutionMode::GPU> cudaRenderer(sharedDimensions, samplesPerPixel, maxDepth, colorMul);
		Renderer<ExecutionMode::CPU> cpuRenderer(sharedDimensions, samplesPerPixel, maxDepth, colorMul);

		sf::Sprite sprite(texture);

		while (isRunning.load(std::memory_order_relaxed))
		{
			if (executionMode.load(std::memory_order_consume) == ExecutionMode::CPU)
			{
				timeTaken = cpuRenderer.Render(sharedDimensions, image, moveCamera).count();
				texture.update(image);
			}
			else
			{
				cudaSurfaceObject_t surface = cudaSurface.BeginFrame();
				timeTaken					= cudaRenderer.Render(sharedDimensions, surface, moveCamera).count();
				cudaSurface.EndFrame(surface);
			}

			window.draw(sprite);
			window.display();

			// Resize texture if dimensions have changed
			if (const auto dims = sharedDimensions.load(std::memory_order_acquire); dims.x != texture.getSize().x || dims.y != texture.getSize().y)
			{
				cudaSurface.UnregisterSurface();
				CHECK_BOOL(texture.resize(dims));
				image.resize(dims);
				sprite.setTexture(texture, true);
				window.setView(sf::View(sf::FloatRect({ 0.f, 0.f }, (sf::Vector2f)dims)));
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
				isRunning = false;
				break;
			}
			else if (const auto* resized = event->getIf<sf::Event::Resized>())
			{
				//  Signal the rendering thread first with the new dimensions
				sharedDimensions.store({ std::max(resized->size.x, 1u), std::max(resized->size.y, 1u) }, std::memory_order_consume);
			}
			else if (const auto* keyReleased = event->getIf<sf::Event::KeyReleased>())
			{
				if (keyReleased->code == sf::Keyboard::Key::F2)
				{
					ExecutionMode expected = executionMode.load(std::memory_order_acquire);
					ExecutionMode desired;
					do
					{
						desired = (expected == ExecutionMode::GPU) ? ExecutionMode::CPU : ExecutionMode::GPU;
					} while (!executionMode.compare_exchange_weak(expected, desired, std::memory_order_acq_rel));
				}
				else if (keyReleased->code == sf::Keyboard::Key::M)
				{
					moveCamera = !moveCamera;
				}
			}
		}
		auto dims = sharedDimensions.load(std::memory_order_relaxed);
		window.setTitle(fmt::format("Ray Tracing On {2}! Press F2 to toggle execution mode | Press M to move camera | Resolution: {0}x{1} | {2} Time: {3:.3f}ms", dims.x, dims.y, executionMode == ExecutionMode::CPU ? "CPU" : "GPU", timeTaken));
	}

	return executionMode == ExecutionMode::CPU ? image : texture.copyToImage(); // GPU to CPU memory
}

int main()
{
	Log::Init();

	const sf::Image& image	 = Render(1280, 720);
	const bool		 success = image.saveToFile("image.png");
	LOG_CORE_INFO("Saving to file: {}", (success ? "successful" : "failed"));

	Log::ShutDown();
}
