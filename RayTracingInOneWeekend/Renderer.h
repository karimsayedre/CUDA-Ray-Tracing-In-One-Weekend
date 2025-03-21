#pragma once

#include <SFML/Graphics/Image.hpp>

class Renderer
{
  public:
	sf::Image Render(const uint32_t width, const uint32_t height);
};
