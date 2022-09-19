#include "Log.h"
#include <glm/glm.hpp>

#include "Viewer.h"
#include "Renderer.h"
#include <SFML/Graphics.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main()
{
	Log::Init();

	Renderer renderer;
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray Tracing In A Weekend!");

	const sf::Image& image = renderer.Render(WIDTH, HEIGHT);
	sf::Texture texture;
	texture.create(WIDTH, HEIGHT);
	texture.update(image);
	sf::Sprite sprite(texture);


	while (window.isOpen())
	{
		// check all the window's events that were triggered since the last iteration of the loop
		sf::Event event;
		while (window.pollEvent(event))
		{
			// "close requested" event: we close the window
			if (event.type == sf::Event::Closed)
				window.close();
		}

		window.clear(sf::Color::Cyan);
		window.draw(sprite);
		window.display();
	}


	//LOG_FILE(image);
	Log::ShutDownFile();
	LOG_CORE_INFO("Finished Successfully!");
	Viewer::OpenImage();

	Viewer::CloseImage();
	Log::ShutDownLogger();
}
