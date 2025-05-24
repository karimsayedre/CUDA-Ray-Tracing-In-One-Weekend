#include "pch.h"
#include <chrono>
#include <GL/gl.h>

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

// Explicit instantiations
template class Renderer<ExecutionMode::GPU>;
template class Renderer<ExecutionMode::CPU>;

__host__ void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		printf("CUDA error: '%s'(%d) at %s:%d '%s' \n", cudaGetErrorString(result), result, file, line, func);
		// Make sure we call CUDA Device Reset before exiting
		CHECK_CUDA_ERRORS(cudaDeviceReset());
		__debugbreak();
		exit(99);
	}
}

__global__ void RenderSeedsInit(const uint32_t maxX, const uint32_t maxY, uint32_t* randSeeds)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY))
		return;
	const uint32_t pixelIndex = j * maxX + i;
	randSeeds[pixelIndex]	  = PcgHash(1984 + pixelIndex);
}

__global__ void CreateWorldKernel()
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		CreateWorld();
	}
}

__global__ void RenderKernel()
{
	const RenderParams* __restrict__ params = GetParams();

	const float x = (float)(threadIdx.x + blockIdx.x * blockDim.x);
	const float y = (float)(threadIdx.y + blockIdx.y * blockDim.y);
	if ((x >= params->ResolutionInfo.z) || (y >= params->ResolutionInfo.w))
		return;

	uint32_t seed = params->RandSeeds[uint32_t(y * params->ResolutionInfo.z + x)];

	Vec3 pixelColor(0.0f);

	for (uint32_t s = 0; s < params->m_SamplesPerPixel; s++)
	{
		const float2 uv	 = float2 { (x + RandomFloat(seed)) * params->ResolutionInfo.x, 1.0f - (y + RandomFloat(seed)) * params->ResolutionInfo.y };
		const Ray	 ray = reinterpret_cast<const Camera&>(params->Camera).GetRay(uv);
		pixelColor += RayColor(ray, seed);
	}

	params->RandSeeds[uint32_t(y * params->ResolutionInfo.z + x)] = seed;
	pixelColor *= params->m_ColorMul;
	pixelColor = glm::sqrt(pixelColor);

	// Convert to uchar4 format
	const uchar4 pixel = make_uchar4(static_cast<uint8_t>(pixelColor.x * 255.f), static_cast<uint8_t>(pixelColor.y * 255.f), static_cast<uint8_t>(pixelColor.z * 255.f), 255);

	// Write to surface
	surf2Dwrite(pixel, params->Image, x * sizeof(uchar4), y);
}

template<ExecutionMode Mode>
__host__ Renderer<Mode>::Renderer(const sf::Vector2u dims, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul)
	: m_SamplesPerPixel(samplesPerPixel), m_MaxDepth(maxDepth), m_ColorMul(colorMul),
	  m_Camera(Vec3(13.0f, 2.0f, 3.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 20.0f, static_cast<float>(dims.x) / static_cast<float>(dims.y))
{
	if constexpr (Mode == ExecutionMode::GPU)
	{
		CHECK_CUDA_ERRORS(cudaDeviceSetLimit(cudaLimitStackSize, 16000));
		CHECK_CUDA_ERRORS(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

		//  Create CUDA events for timing
		for (auto& f : m_Frames)
		{
			CHECK_CUDA_ERRORS(cudaEventCreate(&f.Start));
			CHECK_CUDA_ERRORS(cudaEventCreate(&f.End));
		}
	}

	constexpr int numHitables = 22 * 22 + 1 + 3;

	dp_List		 = Hitables::Init<Mode>(numHitables);
	dp_Materials = Mat::Init<Mode>(numHitables);
	dp_BVH		 = BVH::Init<Mode>(numHitables * 2 - 1);

	CopyDeviceData(0);

	if constexpr (Mode == ExecutionMode::GPU)
	{
		CHECK_CUDA_ERRORS(cudaEventRecord(m_Frames[0].Start));
		CreateWorldKernel<<<1, 1>>>();
		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaEventRecord(m_Frames[0].End));

		CHECK_CUDA_ERRORS(cudaEventSynchronize(m_Frames[0].End));

		float elapsedTimeMs;
		CHECK_CUDA_ERRORS(cudaEventElapsedTime(&elapsedTimeMs, m_Frames[0].Start, m_Frames[0].End));
		printf("CUDA BVH creation took: %.3f ms on GPU\n", elapsedTimeMs);

		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	}
	else
	{
		auto start = std::chrono::high_resolution_clock::now();
		CreateWorld();
		auto end = std::chrono::high_resolution_clock::now();
		printf("CPU BVH creation took %.3f ms on CPU\n", std::chrono::duration<float, std::milli>(end - start).count());
	}

	ResizeImage(dims, 0);
}

template<ExecutionMode Mode>
__host__ void Renderer<Mode>::ResizeImage(const sf::Vector2u dims, cudaSurfaceObject_t surface)
{
	if (m_Dims == dims)
		return;
	m_Dims = dims;

	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	MemPolicy<Mode>::template Resize(dp_RandSeeds, m_Dims.x * m_Dims.y);

	// Update camera with new aspect ratio
	m_Camera.ResizeViewport(float(m_Dims.x) / float(m_Dims.y));

	if constexpr (Mode == ExecutionMode::GPU)
	{
		// Set up grid and block dimensions
		dim3 block(8, 8);
		dim3 grid((m_Dims.x + block.x - 1) / block.x, (m_Dims.y + block.y - 1) / block.y);
		RenderSeedsInit<<<grid, block>>>(m_Dims.x, m_Dims.y, dp_RandSeeds);
		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	}
	else
	{
		// CPU version
		for (uint32_t j = 0; j < m_Dims.y; j++)
		{
			for (uint32_t i = 0; i < m_Dims.x; i++)
			{
				const uint32_t pixelIndex = j * m_Dims.x + i;
				dp_RandSeeds[pixelIndex]  = PcgHash(1984 + pixelIndex);
			}
		}
	}

	CopyDeviceData(surface);
}

template<>
template<>
__host__ std::chrono::duration<float, std::milli> Renderer<ExecutionMode::GPU>::Render(const sf::Vector2u& size, cudaSurfaceObject_t& surface, const bool moveCamera)
{
	while (m_SubmittedCount - m_CompletedCount >= kFramesInFlight)
	{
		// wait for the oldest outstanding frame to finish:
		uint32_t oldestIdx = m_CompletedCount % kFramesInFlight;
		CHECK_CUDA_ERRORS(cudaEventSynchronize(m_Frames[oldestIdx].End));
		// optionally read its timing here…
		m_CompletedCount++;
	}

	if (moveCamera)
		m_Camera.MoveAndLookAtSamePoint({ 0.1f, 0.0f, 0.f }, 10.0f);

	CopyDeviceData(surface);

	dim3 block(8, 8);
	dim3 grid((size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);

	// record & launch into current slot
	int	  idx	= m_SubmittedCount % kFramesInFlight;
	auto& frame = m_Frames[idx];
	CHECK_CUDA_ERRORS(cudaEventRecord(frame.Start));
	RenderKernel<<<grid, block>>>();
	CHECK_CUDA_ERRORS(cudaEventRecord(frame.End));

	m_SubmittedCount++;

	// Optionally, query the *next* completed slot for your timing
	float elapsedMs = 0.f;
	if (m_SubmittedCount > kFramesInFlight)
	{
		int queryIdx = (m_CompletedCount) % kFramesInFlight;
		if (cudaEventQuery(m_Frames[queryIdx].End) == cudaSuccess)
		{
			cudaEventElapsedTime(&elapsedMs,
								 m_Frames[queryIdx].Start,
								 m_Frames[queryIdx].End);
			m_CompletedCount++;
		}
	}
	return std::chrono::duration<float, std::milli>(elapsedMs > 0.f ? elapsedMs : 0.f);
}

template<ExecutionMode Mode>
__host__ void Renderer<Mode>::CopyDeviceData(const cudaSurfaceObject_t imageSurface) const
{
	RenderParams* h_Params		= GetParams();
	h_Params->Image				= imageSurface;
	h_Params->BVH				= dp_BVH;
	h_Params->List				= dp_List;
	h_Params->Materials			= dp_Materials;
	h_Params->ResolutionInfo	= float4 { 1.0f / (float)m_Dims.x, 1.0f / (float)m_Dims.y, (float)m_Dims.x, (float)m_Dims.y };
	h_Params->Camera			= reinterpret_cast<CameraPOD&>(m_Camera);
	h_Params->RandSeeds			= dp_RandSeeds;
	h_Params->m_ColorMul		= m_ColorMul;
	h_Params->m_MaxDepth		= m_MaxDepth;
	h_Params->m_SamplesPerPixel = m_SamplesPerPixel;

	if constexpr (Mode == ExecutionMode::GPU)
		CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_Params, h_Params, sizeof(RenderParams)));
}

template<ExecutionMode Mode>
__host__ Renderer<Mode>::~Renderer()
{
	if constexpr (Mode == ExecutionMode::GPU)
	{
		// Cleanup events
		for (auto& f : m_Frames)
		{
			CHECK_CUDA_ERRORS(cudaEventDestroy(f.Start));
			CHECK_CUDA_ERRORS(cudaEventDestroy(f.End));
		}

		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	}

	if (dp_RandSeeds)
		MemPolicy<Mode>::Free(dp_RandSeeds);
	if (dp_List)
		MemPolicy<Mode>::Free(dp_List);
	if (dp_BVH)
		MemPolicy<Mode>::Free(dp_BVH);
	if (dp_Materials)
		MemPolicy<Mode>::Free(dp_Materials);
}
