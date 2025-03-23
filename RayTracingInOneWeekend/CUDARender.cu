#include "pch.cuh"
#include "BVH.h"
#include "Cuda.h"
#include "HittableList.h"
#include "Material.h"
#include "Random.h"
#include "Raytracer.h"

__host__ void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		printf("CUDA error = %s at %s:%d '%s' \n", cudaGetErrorString(result), file, line, func);
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void RenderSeedsInit(const uint32_t maxX, const uint32_t maxY, uint32_t* randState)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY))
		return;
	uint32_t pixel_index   = j * maxX + i;
	randState[pixel_index] = PcgHash(1984 + pixel_index);
}

CudaRenderer::CudaRenderer(const uint32_t width, const uint32_t height, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul)
	: m_Width(width), m_Height(height), m_SamplesPerPixel(samplesPerPixel), m_MaxDepth(maxDepth), m_ColorMul(colorMul),
	  m_Camera(Vec3(13.0f, 2.0f, 3.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 20.0f, float(width) / float(height))
{
	cudaDeviceSetLimit(cudaLimitStackSize, 2048);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// allocate random seeds
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_RandSeeds, m_Width * m_Height * sizeof(uint32_t)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_Camera, sizeof(Camera)));

	constexpr int numHitables = 22 * 22 + 1 + 3;

	HittableList::Init(d_List, numHitables);
	BVHSoA::Init(d_World, numHitables * 2 - 1);
	Materials::Init(d_Materials, numHitables);

	dim3 block(8, 8);
	dim3 grid((m_Width + block.x - 1) / block.x, (m_Height + block.y - 1) / block.y);

	RenderSeedsInit<<<grid, block>>>(m_Width, m_Height, d_RandSeeds); 
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	CreateWorld<<<1, 1>>>(d_List, d_Materials, d_World);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	cudaDeviceSetLimit(cudaLimitStackSize, 2000);

	d_Image = SetupFramebufferSurface(m_Width, m_Height);
}

__host__ void CudaRenderer::Render() const
{
	dim3 block(8, 8);
	dim3 grid((m_Width + block.x - 1) / block.x, (m_Height + block.y - 1) / block.y);

	// m_Camera.Translate({0.1f, 0.0f, 0.f}, 10.0f);

	CHECK_CUDA_ERRORS(cudaMemcpy(d_Camera, &m_Camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	// Render our buffer
	const clock_t start = clock();
	InternalRender<<<grid, block>>>(d_Image, d_World, d_List, d_Materials, m_Width, m_Height, d_Camera, m_SamplesPerPixel, m_ColorMul, m_MaxDepth, d_RandSeeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	const clock_t stop		   = clock();
	const double  timerSeconds = stop - start;
	std::cerr << "took " << timerSeconds << "ms.\n";
}

CudaRenderer::~CudaRenderer()
{
	cudaFree(d_RandSeeds);
	cudaFree(d_List);
	cudaFree(d_World);
	cudaFree(d_Materials);
	cudaFree(d_Camera);
}
