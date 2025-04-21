#include "pch.h"
#include <chrono>
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
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY))
		return;
	const uint32_t pixelIndex = j * maxX + i;
	randSeeds[pixelIndex]	  = PcgHash(1984 + pixelIndex);
}

__global__ void CreateWorld()
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState localRandState;
		curand_init(1984, 0, 0, &localRandState);

		// Ground sphere:
		Mat::Add(Mat::MaterialType::Lambert, Vec3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f);
		Hitables::Add(Vec3(0, -1000.0f, -1), 1000.0f);

		// For each grid position:
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
#define RND (curand_uniform(&localRandState))

				const float chooseMat = RND;
				const Vec3	center(a + RND, 0.2f, b + RND);
				if (chooseMat < 0.8f)
				{
					Mat::Add(Mat::MaterialType::Lambert, Vec3(RND * RND, RND * RND, RND * RND), 0.0f, 1.0f);
					Hitables::Add(center, 0.2f);
				}
				else if (chooseMat < 0.95f)
				{
					Mat::Add(Mat::MaterialType::Metal, Vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND, 1.0f);
					Hitables::Add(center, 0.2f);
				}
				else
				{
					Mat::Add(Mat::MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
					Hitables::Add(center, 0.2f);
				}
#undef RND
			}
		}

		// Add the three big spheres:
		Mat::Add(Mat::MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
		Hitables::Add(Vec3(0, 1, 0), 1.0f);

		Mat::Add(Mat::MaterialType::Lambert, Vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f);
		Hitables::Add(Vec3(-4, 1, 0), 1.0f);

		Mat::Add(Mat::MaterialType::Metal, Vec3(0.7f, 0.6f, 0.5f), 0.0f, 1.0f);
		Hitables::Add(Vec3(4, 1, 0), 1.0f);

		uint16_t* indices = static_cast<uint16_t*>(malloc(d_Params.List->Count * sizeof(uint16_t)));
		for (uint16_t index = 0; index < d_Params.List->Count; ++index)
			indices[index] = index;

		d_Params.BVH->m_Root = BVH::Build(indices, 0, d_Params.List->Count);

		// d_Params.BVH->m_Root = BVH::ReorderBVH(); // Disabled for now

		free(indices);

		printf("BVHNode created with %u nodes.\n", d_Params.BVH->m_Count);
		printf("BVHNode Root: %u\n", d_Params.BVH->m_Root);
		// DebugBVHNode(d_World, d_World->root);
	}
}

__device__ Vec3 RayColor(const Ray& ray, uint32_t& randSeed)
{
	Ray	 origRay = ray;
	Vec3 curAttenuation(1.0f);

	for (uint32_t i = 0; i < d_Params.m_MaxDepth; i++)
	{
		HitRecord rec;
		// Early exit with sky color if no hit
		if (!BVH::Traverse(origRay, 0.001f, FLT_MAX, rec))
		{
			// Streamlined sky color calculation
			const float t = (0.5f) * (origRay.Direction.y + 1.0f);
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
		if (!Mat::Scatter(origRay, rec, curAttenuation, randSeed))
			return curAttenuation;
	}

	return curAttenuation; // Exceeded max depth
}

__global__ void RenderKernel()
{
	const float x = (float)(threadIdx.x + blockIdx.x * blockDim.x);
	const float y = (float)(threadIdx.y + blockIdx.y * blockDim.y);
	if ((x >= d_Params.ResolutionInfo.z) || (y >= d_Params.ResolutionInfo.w))
		return;

	const uint32_t pixelIndex = y * d_Params.ResolutionInfo.z + x;

	uint32_t seed = d_Params.RandSeeds[pixelIndex];

	Vec3 pixelColor(0.0f);

	for (uint32_t s = 0; s < d_Params.m_SamplesPerPixel; s++)
	{
		const float2 uv = float2 {(x + RandomFloat(seed)) * d_Params.ResolutionInfo.x, 1.0f - (y + RandomFloat(seed)) * d_Params.ResolutionInfo.y};

		const auto ray = reinterpret_cast<Camera&>(d_Params.Camera).GetRay(uv);

		pixelColor += RayColor(ray, seed);
	}

	d_Params.RandSeeds[pixelIndex] = seed;
	pixelColor *= d_Params.m_ColorMul;
	pixelColor = glm::sqrt(pixelColor);

	// Convert to uchar4 format
	const uchar4 pixel = make_uchar4(static_cast<uint8_t>(pixelColor.x * 255.f), static_cast<uint8_t>(pixelColor.y * 255.f), static_cast<uint8_t>(pixelColor.z * 255.f), 255);

	// Write to surface
	surf2Dwrite(pixel, d_Params.Image, x * sizeof(uchar4), y);
}

__host__ CudaRenderer::CudaRenderer(const sf::Vector2u dims, const uint32_t samplesPerPixel, const uint32_t maxDepth, const float colorMul)
	: m_SamplesPerPixel(samplesPerPixel), m_MaxDepth(maxDepth), m_ColorMul(colorMul),
	  m_Camera(Vec3(13.0f, 2.0f, 3.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 20.0f, static_cast<float>(dims.x), static_cast<float>(dims.y))
{
	CHECK_CUDA_ERRORS(cudaDeviceSetLimit(cudaLimitStackSize, 16000));
	CHECK_CUDA_ERRORS(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	//  Create CUDA events for timing
	CHECK_CUDA_ERRORS(cudaEventCreate(&m_StartEvent));
	CHECK_CUDA_ERRORS(cudaEventCreate(&m_EndEvent));

	constexpr int numHitables = 22 * 22 + 1 + 3;

	dp_List		 = Hitables::Init(numHitables);
	dp_Materials = Mat::Init(numHitables);
	dp_BVH		 = BVH::Init(numHitables * 2 - 1);

	CopyDeviceData();

	CreateWorld<<<1, 1>>>();
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	ResizeImage(dims);
}

void CudaRenderer::ResizeImage(const sf::Vector2u dims)
{
	if (m_Dims == dims)
		return;
	ReleaseResizables();

	m_Dims = dims;

	// Ensure all prior operations are complete
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&dp_RandSeeds, m_Dims.x * m_Dims.y * sizeof(uint32_t)));

	// Allocate CUDA array
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	CHECK_CUDA_ERRORS(cudaMallocArray(&d_ImageArray, &channelDesc, m_Dims.x, m_Dims.y, cudaArraySurfaceLoadStore));

	// Create surface object
	cudaResourceDesc resDesc;
	resDesc.resType			= cudaResourceTypeArray;
	resDesc.res.array.array = d_ImageArray;

	CHECK_CUDA_ERRORS(cudaCreateSurfaceObject(&dp_Image, &resDesc));

	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	// Update camera with new aspect ratio
	m_Camera.ResizeViewport(float(m_Dims.x) / float(m_Dims.y));

	// Set up grid and block dimensions
	dim3 block(8, 8);
	dim3 grid((m_Dims.x + block.x - 1) / block.x, (m_Dims.y + block.y - 1) / block.y);
	RenderSeedsInit<<<grid, block>>>(m_Dims.x, m_Dims.y, dp_RandSeeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	CopyDeviceData();
}

__host__ std::chrono::duration<float, std::milli> CudaRenderer::Render(const uint32_t width, const uint32_t height) const
{
	dim3 block(8, 8);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// m_Camera.MoveAndLookAtSamePoint({0.1f, 0.0f, 0.f}, 10.0f);

	CopyDeviceData();

	// Record start event and launch kernel
	CHECK_CUDA_ERRORS(cudaEventRecord(m_StartEvent));
	RenderKernel<<<grid, block>>>();
	CHECK_CUDA_ERRORS(cudaGetLastError()); // Check for launch errors

	// Record end event and synchronize
	CHECK_CUDA_ERRORS(cudaEventRecord(m_EndEvent));
	CHECK_CUDA_ERRORS(cudaEventSynchronize(m_EndEvent));

	// Calculate elapsed time
	float elapsedTimeMs;
	CHECK_CUDA_ERRORS(cudaEventElapsedTime(&elapsedTimeMs, m_StartEvent, m_EndEvent));

	return std::chrono::duration<float, std::milli>(elapsedTimeMs);
}

void CudaRenderer::CopyDeviceData() const
{
	RenderParams h_params;
	h_params.Image			   = dp_Image;
	h_params.BVH			   = dp_BVH;
	h_params.List			   = dp_List;
	h_params.Materials		   = dp_Materials;
	h_params.ResolutionInfo	   = float4 {1.0f / (float)m_Dims.x, 1.0f / (float)m_Dims.y, (float)m_Dims.x, (float)m_Dims.y};
	h_params.Camera			   = reinterpret_cast<CameraPOD&>(m_Camera);
	h_params.RandSeeds		   = dp_RandSeeds;
	h_params.m_ColorMul		   = m_ColorMul;
	h_params.m_MaxDepth		   = m_MaxDepth;
	h_params.m_SamplesPerPixel = m_SamplesPerPixel;

	CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_Params, &h_params, sizeof(RenderParams)));
}

void CudaRenderer::ReleaseResizables() const
{
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(dp_Image));
	if (dp_RandSeeds)
		CHECK_CUDA_ERRORS(cudaFree(dp_RandSeeds));
}

__host__ CudaRenderer::~CudaRenderer()
{
	// Cleanup events
	CHECK_CUDA_ERRORS(cudaEventDestroy(m_StartEvent));
	CHECK_CUDA_ERRORS(cudaEventDestroy(m_EndEvent));

	ReleaseResizables();
	if (dp_List)
		CHECK_CUDA_ERRORS(cudaFree(dp_List));
	if (dp_BVH)
		CHECK_CUDA_ERRORS(cudaFree(dp_BVH));
	if (dp_Materials)
		CHECK_CUDA_ERRORS(cudaFree(dp_Materials));
}
