#include "pch.cuh"

#include <iostream>

#include "Log.h"

#include "Viewer.h"
#include "Renderer.h"


constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main()
{
	Log::Init();

	Renderer renderer;

	const sf::Image image = renderer.Render(WIDTH, HEIGHT);
	
	LOG_CORE_INFO("Saving to file: {}", (image.saveToFile("image.png") ? "successful" : "failed"));

	Log::ShutDownFile();
	LOG_CORE_INFO("Finished Successfully!");
	Viewer::OpenImage();

	Viewer::CloseImage();
	Log::ShutDownLogger();

	//std::cin.get();
}
