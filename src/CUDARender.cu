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
#include <omp.h>

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
		CreateWorld<GpuRNG>();
	}
}

__global__ void RenderKernel()
{
	const auto* params = GetParams();

	const float x = (float)(threadIdx.x + blockIdx.x * blockDim.x);
	const float y = (float)(threadIdx.y + blockIdx.y * blockDim.y);
	if ((x >= params->ResolutionInfo.z) || (y >= params->ResolutionInfo.w))
		return;

	const uint32_t pixelIndex = y * params->ResolutionInfo.z + x;

	uint32_t seed = params->RandSeeds[pixelIndex];

	Vec3 pixelColor(0.0f);

	for (uint32_t s = 0; s < params->m_SamplesPerPixel; s++)
	{
		const float2 uv = float2 {(x + RandomFloat(seed)) * params->ResolutionInfo.x, 1.0f - (y + RandomFloat(seed)) * params->ResolutionInfo.y};

		const auto ray = reinterpret_cast<const Camera&>(params->Camera).GetRay(uv);

		pixelColor += RayColor(ray, seed);
	}

	params->RandSeeds[pixelIndex] = seed;
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
	  m_Camera(Vec3(13.0f, 2.0f, 3.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 20.0f, static_cast<float>(dims.x), static_cast<float>(dims.y))
{
	if constexpr (Mode == ExecutionMode::GPU)
	{
		CHECK_CUDA_ERRORS(cudaDeviceSetLimit(cudaLimitStackSize, 16000));
		CHECK_CUDA_ERRORS(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

		//  Create CUDA events for timing
		CHECK_CUDA_ERRORS(cudaEventCreate(&m_StartEvent));
		CHECK_CUDA_ERRORS(cudaEventCreate(&m_EndEvent));
	}

	constexpr int numHitables = 22 * 22 + 1 + 3;

	dp_List		 = Hitables::Init<Mode>(numHitables);
	dp_Materials = Mat::Init<Mode>(numHitables);
	dp_BVH		 = BVH::Init<Mode>(numHitables * 2 - 1);

	// if constexpr (Mode == ExecutionMode::GPU)
	CopyDeviceData(0);

	if constexpr (Mode == ExecutionMode::GPU)
	{
		CreateWorldKernel<<<1, 1>>>();
		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	}
	else
	{
		CreateWorld<CpuRNG>();
	}

	ResizeImage(dims, 0);
}

template<ExecutionMode Mode>
void Renderer<Mode>::ResizeImage(const sf::Vector2u dims, cudaSurfaceObject_t surface)
{
	if (m_Dims == dims)
		return;

	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	if (dp_RandSeeds)
		MemPolicy<Mode>::Free(dp_RandSeeds);

	m_Dims = dims;

	dp_RandSeeds = MemPolicy<Mode>::template Alloc<uint32_t>(m_Dims.x * m_Dims.y);

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
std::chrono::duration<float, std::milli> Renderer<ExecutionMode::GPU>::Render(const sf::Vector2u& size, cudaSurfaceObject_t& surface)
{
	// m_Camera.MoveAndLookAtSamePoint({0.1f, 0.0f, 0.f}, 10.0f);

	CopyDeviceData(surface);

	dim3 block(8, 8);
	dim3 grid((size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);

	// Record start event and launch kernel
	CHECK_CUDA_ERRORS(cudaEventRecord(m_StartEvent));
	RenderKernel<<<grid, block>>>();
	CHECK_CUDA_ERRORS(cudaGetLastError()); // Check for launch errors
	CHECK_CUDA_ERRORS(cudaEventRecord(m_EndEvent));

	// Record end event and synchronize
	CHECK_CUDA_ERRORS(cudaEventSynchronize(m_EndEvent));

	// Calculate elapsed time
	float elapsedTimeMs;
	CHECK_CUDA_ERRORS(cudaEventElapsedTime(&elapsedTimeMs, m_StartEvent, m_EndEvent));
	return std::chrono::duration<float, std::milli>(elapsedTimeMs);
}

template<ExecutionMode Mode>
void Renderer<Mode>::CopyDeviceData(const cudaSurfaceObject_t imageSurface) const
{
	auto* h_Params				= GetParams();
	h_Params->Image				= imageSurface;
	h_Params->BVH				= dp_BVH;
	h_Params->List				= dp_List;
	h_Params->Materials			= dp_Materials;
	h_Params->ResolutionInfo	= float4 {1.0f / (float)m_Dims.x, 1.0f / (float)m_Dims.y, (float)m_Dims.x, (float)m_Dims.y};
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
		CHECK_CUDA_ERRORS(cudaEventDestroy(m_StartEvent));
		CHECK_CUDA_ERRORS(cudaEventDestroy(m_EndEvent));

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
