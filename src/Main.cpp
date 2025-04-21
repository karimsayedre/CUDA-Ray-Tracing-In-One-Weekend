#include "pch.h"
#include "Log.h"
#include "Renderer.h"
sf::Image image; // Global to prevent copy when returning from Render function

int main()
{
	Log::Init();

	Render(1280, 720);

	const bool success = image.saveToFile("image.png");
	LOG_CORE_INFO("Saving to file: {}", (success ? "successful" : "failed"));

	Log::ShutDown();
}
