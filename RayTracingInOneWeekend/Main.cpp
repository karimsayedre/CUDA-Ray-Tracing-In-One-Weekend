#include "pch.cuh"
#include "Log.h"
#include "Renderer.h"

int main()
{
	Log::Init();

	constexpr int	WIDTH  = 1280;
	constexpr int	HEIGHT = 720;
	const sf::Image image  = Render(WIDTH, HEIGHT);

	LOG_CORE_INFO("Saving to file: {}", (image.saveToFile("image.png") ? "successful" : "failed"));

	LOG_CORE_INFO("Finished Successfully!");
	Log::ShutDown();

	// std::cin.get();
}
