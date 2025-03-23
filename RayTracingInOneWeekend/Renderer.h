#pragma once

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <SFML/System/Thread.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics/Image.hpp>
#include "CudaRenderer.cuh"

inline sf::Image Render(const uint32_t width, const uint32_t height)
{
	// Image
	constexpr int	samplesPerPixel = 30;
	constexpr float colorMul		= 1.0f / (float)samplesPerPixel;
	constexpr int	maxDepth		= 50;

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

			sf::Event event;
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
				{
					window.close();
					isRunning = false;
					break;
				}
			}

			texture.loadFromImage(image);
			window.draw(sprite);

			window.display();
		}
	});
	OpenGLThread.launch();

	CudaRenderer cudaRenderer(width, height, samplesPerPixel, maxDepth, colorMul);
	while (isRunning)
	{
		cudaRenderer.Render();

		cudaResourceDesc resDesc;
		cudaGetSurfaceObjectResourceDesc(&resDesc, cudaRenderer.GetDeviceImage());
		cudaArray_t cuArray = resDesc.res.array.array;
		cudaMemcpy2DFromArray((void*)image.getPixelsPtr(), width * 4, cuArray, 0, 0, width * 4, height, cudaMemcpyDeviceToHost);

		{
			//  Notify the OpenGL thread a new frame is ready:
			std::unique_lock<std::mutex> lock(frameMutex);
			frameReady = true;
		}
		frameCV.notify_one();
	}

	OpenGLThread.wait();

	return image;
}
