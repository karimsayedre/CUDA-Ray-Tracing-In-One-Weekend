#include <chrono>

#include "pch.cuh"
#include "BVH.h"
#include "Cuda.h"
#include "HittableList.h"
#include "Material.h"
#include "Random.h"

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
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY))
		return;
	const uint32_t pixelIndex = j * maxX + i;
	randSeeds[pixelIndex]	  = PcgHash(1984 + pixelIndex);
}

__global__ void CreateWorld(Hitables::HittableList* d_List, Mat::Materials* d_Materials, BVH::BVHSoA* d_World)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState localRandState;
		curand_init(1984, 0, 0, &localRandState);

		// Ground sphere:
		Add(d_Materials, Mat::MaterialType::Lambert, Vec3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f);
		Hitables::Add(d_List, Vec3(0, -1000.0f, -1), 1000.0f);

		// For each grid position:
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
#define RND (curand_uniform(&localRandState))

				const float chooseMat = RND;
				Vec3		center(a + RND, 0.2f, b + RND);
				if (chooseMat < 0.8f)
				{
					Add(d_Materials, Mat::MaterialType::Lambert, Vec3(RND * RND, RND * RND, RND * RND), 0.0f, 1.0f);
					Hitables::Add(d_List, center, 0.2f);
				}
				else if (chooseMat < 0.95f)
				{
					Add(d_Materials, Mat::MaterialType::Metal, Vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND, 1.0f);
					Hitables::Add(d_List, center, 0.2f);
				}
				else
				{
					Add(d_Materials, Mat::MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
					Hitables::Add(d_List, center, 0.2f);
				}
#undef RND
			}
		}

		// Add the three big spheres:
		Mat::Add(d_Materials, Mat::MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
		Hitables::Add(d_List, Vec3(0, 1, 0), 1.0f);

		Mat::Add(d_Materials, Mat::MaterialType::Lambert, Vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f);
		Hitables::Add(d_List, Vec3(-4, 1, 0), 1.0f);

		Mat::Add(d_Materials, Mat::MaterialType::Metal, Vec3(0.7f, 0.6f, 0.5f), 0.0f, 1.0f);
		Hitables::Add(d_List, Vec3(4, 1, 0), 1.0f);

		uint16_t* indices = static_cast<uint16_t*>(malloc(d_List->Count * sizeof(uint16_t)));
		for (uint16_t index = 0; index < d_List->Count; ++index)
			indices[index] = index;

		d_World->m_Root = BVH::Build(d_List, indices, 0, d_List->Count, d_World);
		printf("BVH created with %u nodes.\n", d_World->m_Count);

		printf("BVH Root: %u\n", d_World->m_Root);
		// DebugBVHNode(d_World, d_World->root);

		free(indices);
	}
}

__device__ Vec3 RayColor(Ray& ray, BVH::BVHSoA* __restrict__ d_World, Hitables::HittableList* __restrict__ d_List, Mat::Materials* __restrict__ d_Materials, const uint32_t depth, uint32_t& randSeed)
{
	Vec3 curAttenuation(1.0f);

	for (uint32_t i = 0; i < depth; i++)
	{
		HitRecord rec;
		// Early exit with sky color if no hit
		if (!Traverse(ray, 0.001f, FLT_MAX, d_List, rec, d_World))
		{
			// Streamlined sky color calculation
			const float t = (0.5f) * (ray.Direction().y + 1.0f);
			return curAttenuation * ((1.0f - t) * 1.0f + t * Vec3(0.5f, 0.7f, 1.0f));
		}

		// if (i > 3)
		{
			const float rrProb = glm::max(curAttenuation.x, glm::max(curAttenuation.y, curAttenuation.z));

			// Early termination with zero - saves registers by avoiding division
			if (RandomFloat(randSeed) > rrProb)
				return Vec3(0.0f);

			// Apply RR adjustment directly to current attenuation
			curAttenuation /= rrProb;
		}

		// Early exit on no scatter
		if (!Mat::Scatter(d_Materials, ray, rec, curAttenuation, randSeed))
			return curAttenuation;
	}

	return curAttenuation; // Exceeded max depth
}

// Modified kernel using surface object
__global__ void InternalRender(cudaSurfaceObject_t fb, BVH::BVHSoA* __restrict__ d_World, Hitables::HittableList* __restrict__ d_List, Mat::Materials* __restrict__ d_Materials, const uint32_t maxX, const uint32_t maxY, Camera* camera, uint32_t samplersPerPixel, float colorMul, uint32_t maxDepth, uint32_t* randSeeds)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= maxX) || (y >= maxY))
		return;

	uint32_t pixelIndex = y * maxX + x;

	uint32_t seed = randSeeds[pixelIndex];
	Vec3	 pixelColor(0.0f);
	for (uint32_t s = 0; s < samplersPerPixel; s++)
	{
		float u	 = ((float)x + RandomFloat(seed)) / (float)maxX;
		float v	 = ((float)y + RandomFloat(seed)) / (float)maxY;
		v		 = 1.0f - v;
		auto ray = camera->GetRay(u, v);

		pixelColor += RayColor(ray, d_World, d_List, d_Materials, maxDepth, seed);
	}
	randSeeds[pixelIndex] = seed;
	pixelColor *= colorMul;
	pixelColor = glm::sqrt(pixelColor);

	// Convert to uchar4 format
	uchar4 pixel = make_uchar4(
		static_cast<unsigned char>(pixelColor.x * 255.f),
		static_cast<unsigned char>(pixelColor.y * 255.f),
		static_cast<unsigned char>(pixelColor.z * 255.f),
		255);

	// Write to surface
	surf2Dwrite(pixel, fb, x * sizeof(uchar4), y);
}

__host__ CudaRenderer::CudaRenderer(const Dimensions dims, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul)
	: m_SamplesPerPixel(samplesPerPixel), m_MaxDepth(maxDepth), m_ColorMul(colorMul),
	  m_Camera(Vec3(13.0f, 2.0f, 3.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 20.0f, static_cast<float>(dims.Width) / static_cast<float>(dims.Height))
{
	CHECK_CUDA_ERRORS(cudaDeviceSetLimit(cudaLimitStackSize, 16000));
	CHECK_CUDA_ERRORS(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// allocate random seeds
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_Camera, sizeof(Camera)));

	constexpr int numHitables = 22 * 22 + 1 + 3;

	Hitables::InitHittableList(d_List, numHitables);
	Mat::InitMaterials(d_Materials, numHitables);
	BVH::InitBVH(d_World, numHitables * 2 - 1);

	CreateWorld<<<1, 1>>>(d_List, d_Materials, d_World);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	ResizeImage(dims.Width, dims.Height);
}

void CudaRenderer::ResizeImage(const uint32_t width, const uint32_t height)
{
	// Ensure all prior operations are complete
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_RandSeeds, width * height * sizeof(uint32_t)));

	// Set up grid and block dimensions
	dim3 block(8, 8);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// Initialize random seeds
	RenderSeedsInit<<<grid, block>>>(width, height, d_RandSeeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	// Allocate CUDA array
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	CHECK_CUDA_ERRORS(cudaMallocArray(&d_ImageArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

	// Create surface object
	cudaResourceDesc resDesc;
	resDesc.resType			= cudaResourceTypeArray;
	resDesc.res.array.array = d_ImageArray;

	CHECK_CUDA_ERRORS(cudaCreateSurfaceObject(&d_Image, &resDesc));

	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	// Update camera with new aspect ratio
	m_Camera.ResizeViewport(float(width) / float(height));

	m_Camera = Camera(Vec3(13.0f, 2.0f, 3.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 20.0f, static_cast<float>(width) / static_cast<float>(height));
}

__host__ std::chrono::duration<float, std::milli> CudaRenderer::Render(const uint32_t width, const uint32_t height) const
{
	// Copy camera data to device
	CHECK_CUDA_ERRORS(cudaMemcpy(d_Camera, &m_Camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	dim3 block(8, 8);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// Create CUDA events for timing
	cudaEvent_t startEvent, endEvent;
	CHECK_CUDA_ERRORS(cudaEventCreate(&startEvent));
	CHECK_CUDA_ERRORS(cudaEventCreate(&endEvent));

	// Record start event and launch kernel
	CHECK_CUDA_ERRORS(cudaEventRecord(startEvent));
	InternalRender<<<grid, block>>>(d_Image, d_World, d_List, d_Materials, width, height, d_Camera, m_SamplesPerPixel, m_ColorMul, m_MaxDepth, d_RandSeeds);
	CHECK_CUDA_ERRORS(cudaGetLastError()); // Check for launch errors

	// Record end event and synchronize
	CHECK_CUDA_ERRORS(cudaEventRecord(endEvent));
	CHECK_CUDA_ERRORS(cudaEventSynchronize(endEvent));

	// Calculate elapsed time
	float elapsedTimeMs;
	CHECK_CUDA_ERRORS(cudaEventElapsedTime(&elapsedTimeMs, startEvent, endEvent));

	// Cleanup events
	CHECK_CUDA_ERRORS(cudaEventDestroy(startEvent));
	CHECK_CUDA_ERRORS(cudaEventDestroy(endEvent));

	return std::chrono::duration<float, std::milli>(elapsedTimeMs);
}

__host__ CudaRenderer::~CudaRenderer()
{
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(d_Image));
	CHECK_CUDA_ERRORS(cudaFree(d_RandSeeds));
	CHECK_CUDA_ERRORS(cudaFree(d_List));
	CHECK_CUDA_ERRORS(cudaFree(d_World));
	CHECK_CUDA_ERRORS(cudaFree(d_Materials));
	CHECK_CUDA_ERRORS(cudaFree(d_Camera));
}
