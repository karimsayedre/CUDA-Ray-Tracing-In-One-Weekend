#include "pch.h"
#include <chrono>
#include "BVH.h"
#include "Cuda.h"
#include "HittableList.h"
#include "Material.h"
#include "Random.h"
#include "CPU_GPU.h"
#include "Raytracing.h"

#include "Renderer.h"
#include "ThreadPool.h"
#include "SFML/Graphics/Image.hpp"

RenderParams h_Params; // unified host copy

template<>
template<>
__host__ std::chrono::duration<float, std::milli> Renderer<ExecutionMode::CPU>::Render<sf::Image>(const sf::Vector2u& size, sf::Image& surface, bool moveCamera)
{
	if (moveCamera)
		m_Camera.MoveAndLookAtSamePoint({ 0.1f, 0.0f, 0.f }, 10.0f);

	const std::chrono::high_resolution_clock::time_point start	= std::chrono::high_resolution_clock::now();
	const RenderParams*									 params = GetParams();

	constexpr uint32_t tileWidth  = 16;
	constexpr uint32_t tileHeight = 16;

	CopyDeviceData(0);

	ThreadPool					   pool;
	std::vector<std::future<void>> futures;

	// Enqueue one task per tile
	for (uint32_t tileY = 0; tileY < size.y; tileY += tileHeight)
	{
		for (uint32_t tileX = 0; tileX < size.x; tileX += tileWidth)
		{
			futures.push_back(pool.Enqueue([&, tileX, tileY]()
			{
				const uint32_t xEnd = glm::min(tileX + tileWidth, size.x);
				const uint32_t yEnd = glm::min(tileY + tileHeight, size.y);
				for (uint32_t y = tileY; y < yEnd; ++y)
				{
					for (uint32_t x = tileX; x < xEnd; ++x)
					{
						const uint32_t pixelIndex = y * params->ResolutionInfo.z + x;
						uint32_t	   seed		  = params->RandSeeds[pixelIndex];
						Vec3		   pixelColor { 0.0f };

						for (uint32_t s = 0; s < m_SamplesPerPixel; ++s)
						{
							const float2 uv	 = float2 { (x + RandomFloat(seed)) * params->ResolutionInfo.x, 1.0f - (y + RandomFloat(seed)) * params->ResolutionInfo.y };
							const Ray	 ray = reinterpret_cast<const Camera&>(params->Camera).GetRay(uv);
							pixelColor += RayColor(ray, seed);
						}

						params->RandSeeds[pixelIndex] = seed;
						pixelColor *= m_ColorMul;
						pixelColor = glm::sqrt(pixelColor);

						surface.setPixel({ x, y }, { static_cast<uint8_t>(pixelColor.x * 255.f), static_cast<uint8_t>(pixelColor.y * 255.f), static_cast<uint8_t>(pixelColor.z * 255.f), 255 });
					}
				}
			}));
		}
	}

	// Wait for all tile m_Tasks to complete
	for (auto& f : futures)
		f.get();

	const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	return end - start;
}
