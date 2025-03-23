#include "pch.cuh"
#include "BVH.h"
#include "HittableList.h"
#include "Material.h"
#include "Random.h"

#define RND (curand_uniform(&local_rand_state))

__host__ void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		printf("CUDA error = %s at %s:%d '%s' \n", cudaGetErrorString(result), file, line, func);
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void rand_init(curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(uint32_t max_x, uint32_t max_y, uint32_t* rand_state)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;
	uint32_t pixel_index = j * max_x + i;
	// Original: Each thread gets same seed, a different sequence number, no offset
	// curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
	// BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
	// performance improvement of about 2x!
	// curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);

	rand_state[pixel_index] = pcg_hash(1984 + pixel_index);
}

__global__ void create_world(HittableList* d_list, Materials* d_materials, BVHSoA* d_world, int nx, int ny, curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		uint16_t	i				 = 0;

		// Ground sphere:
		d_materials->Add(MaterialType::Lambert, Vec3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f);
		d_list->Add(Vec3(0, -1000.0f, -1), 1000.0f);
		i++;

		// For each grid position:
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				Float choose_mat = RND;
				Vec3  center(a + RND, 0.2f, b + RND);
				if (choose_mat < __float2half(0.8f))
				{
					d_materials->Add(MaterialType::Lambert, Vec3(RND * RND, RND * RND, RND * RND), 0.0f, 1.0f);
					d_list->Add(center, 0.2f);
					i++;
				}
				else if (choose_mat < __float2half(0.95f))
				{
					d_materials->Add(MaterialType::Metal, Vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND, 1.0f);
					d_list->Add(center, 0.2f);
					i++;
				}
				else
				{
					d_materials->Add(MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
					d_list->Add(center, 0.2f);
					i++;
				}
			}
		}

		// Add the three big spheres:
		d_materials->Add(MaterialType::Dielectric, Vec3(1.0), 0.0f, 1.5f);
		d_list->Add(Vec3(0, 1, 0), 1.0f);

		i++;
		d_materials->Add(MaterialType::Lambert, Vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f);
		d_list->Add(Vec3(-4, 1, 0), 1.0f);
		i++;

		d_materials->Add(MaterialType::Metal, Vec3(0.7f, 0.6f, 0.5f), 0.0f, 1.0f);
		d_list->Add(Vec3(4, 1, 0), 1.0f);
		i++;

		*rand_state = local_rand_state;

		uint16_t* indices = (uint16_t*)malloc(d_list->m_Count * sizeof(uint16_t));
		for (uint32_t index = 0; index < d_list->m_Count; ++index)
			indices[index] = index;

		d_world->root = d_world->BuildBVH_SoA(d_list, indices, 0, d_list->m_Count);
		printf("BVH created with %u nodes.\n", d_world->m_count);

		printf("BVH Root: %u\n", d_world->root);
		// DebugBVHNode(d_world, d_world->root);

		free(indices);
	}
}

__device__ static Vec3 RayColor(Ray& ray, BVHSoA* __restrict__ world, HittableList* __restrict__ list, Materials* __restrict__ materials, const uint32_t depth, uint32_t& randSeed)
{
	Vec3 cur_attenuation(1.0f);

	for (uint32_t i = 0; i < depth; i++)
	{
		HitRecord rec;

		// Early exit with sky color if no hit
		if (!world->TraverseBVH_SoA(ray, 0.001f, FLT_MAX, list, rec))
		{
			// Streamlined sky color calculation
			const Float t = (__float2half(0.5f)) * (ray.Direction().y + __float2half(1.0f));
			return cur_attenuation * ((__float2half(1.0f) - t) * 1.0f + t * Vec3(0.5f, 0.7f, 1.0f));
		}

		// Russian Roulette - simplified
		// Only apply after a few bounces (commented out in original)
		/*if (i > 3)*/ {
			// Use max component for probability
			const Float rrProb = glm::hmax(cur_attenuation.x, glm::hmax(cur_attenuation.y, cur_attenuation.z));

			// Early termination with zero - saves registers by avoiding division
			if (RandomFloat(randSeed) > rrProb)
				return Vec3(0.0f);

			// Apply RR adjustment directly to current attenuation
			cur_attenuation /= rrProb;
		}

		// Early exit on no scatter
		if (!materials->Scatter(ray, rec, cur_attenuation, randSeed))
			return Vec3(0.0f);
	}

	return Vec3(0.0f); // Exceeded max depth
}

// Modified kernel using surface object
__global__ void InternalRender(cudaSurfaceObject_t fb, BVHSoA* __restrict__ world, HittableList* __restrict__ list, Materials* __restrict__ materials, uint32_t max_x, uint32_t max_y, Camera* camera, uint32_t samplersPerPixel, Float colorMul, uint32_t maxDepth, uint32_t* randSeeds)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;

	uint32_t pixel_index = j * max_x + i;

	uint32_t seed = randSeeds[pixel_index];
	Vec3	 pixel_color(0.0f);
	for (uint32_t s = 0; s < samplersPerPixel; s++)
	{
		Float u = Float(Float(i) + RandomFloat(seed)) / Float(max_x);
		Float v = Float(Float(j) + RandomFloat(seed)) / Float(max_y);
		// u = 1.0f - u;
		v		 = __float2half(1.0f) - v;
		auto ray = camera->GetRay(u, v);

		pixel_color += RayColor(ray, world, list, materials, maxDepth, seed);
	}
	randSeeds[pixel_index] = seed;
	pixel_color *= colorMul;
	pixel_color = glm::sqrt(pixel_color);

	// Convert to uchar4 format
	uchar4 pixel = make_uchar4(
		static_cast<unsigned char>(pixel_color.x * __float2half(255.f)),
		static_cast<unsigned char>(pixel_color.y * __float2half(255.f)),
		static_cast<unsigned char>(pixel_color.z * __float2half(255.f)),
		255);

	// Write to surface
	surf2Dwrite(pixel, fb, i * sizeof(uchar4), j);
}

// Host code to set up the surface object
cudaSurfaceObject_t setupFramebufferSurface(uint32_t width, uint32_t height)
{
	// Allocate CUDA array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaArray_t			  cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);

	// Create surface object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType			= cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaSurfaceObject_t surfObj;
	cudaCreateSurfaceObject(&surfObj, &resDesc);

	return surfObj;
}

__host__ void CudaRenderer::Init()
{
	cudaDeviceSetLimit(cudaLimitStackSize, 20000);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	// allocate random state
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_rand_seeds, m_Width * m_Height * sizeof(uint32_t)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	constexpr int numHitables = 22 * 22 + 1 + 3;

	// Allocate memory for the HittableList struct in device memory
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_list, sizeof(HittableList)));

	// Copy the hitable count to the device
	CHECK_CUDA_ERRORS(cudaMemcpy(&(d_list->m_Count), &numHitables, sizeof(int), cudaMemcpyHostToDevice));

	HittableList::Init(d_list, numHitables);
	BVHSoA::Init(d_world, numHitables * 2 - 1);
	Materials::Init(d_materials, numHitables);

	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	// MaterialSoA::Init(numHitables);

	rand_init<<<1, 1>>>(d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	dim3 block(8, 8);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	render_init<<<grid, block>>>(m_Width, m_Height, d_rand_seeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	create_world<<<1, 1>>>(d_list, d_materials, d_world, m_Width, m_Height, d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	cudaDeviceSetLimit(cudaLimitStackSize, 200);

	d_Image = setupFramebufferSurface(m_Width, m_Height);
}

__host__ void CudaRenderer::Render() const
{
	dim3 block(8, 8);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	Float aspectRatio = Float(m_Width) / Float(m_Height);

	static Float distance = 0.0f;
	Camera		 camera(Vec3(__float2half(13.0f) + distance, 2.0f, 3.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 20.0f, aspectRatio);

	CHECK_CUDA_ERRORS(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	// distance += 0.1f;

	if (distance > __float2half(10.0f))
		distance = 0.0f;

	const clock_t start = clock();
	// Render our buffer
	InternalRender<<<grid, block>>>(d_Image, d_world, d_list, d_materials, m_Width, m_Height, d_camera, m_SamplesPerPixel, m_ColorMul, m_MaxDepth, d_rand_seeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	const clock_t stop		   = clock();
	const double  timerSeconds = stop - start;
	std::cerr << "took " << timerSeconds << "ms.\n";
}

CudaRenderer::~CudaRenderer()
{
	cudaFree(d_rand_seeds);
	cudaFree(d_rand_state2);
	cudaFree(d_list);
	// cudaFree(d_materials);
	cudaFree(d_world);
	cudaFree(d_camera);
	// cudaFree(d_Image);
}
