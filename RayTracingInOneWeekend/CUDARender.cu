#include "pch.cuh"

#include <cuda_runtime.h>
#include <future>
#include <cuda.h>
#include "CudaRenderer.cuh"

#include <curand_kernel.h>
#include <mutex>

#include "BVH.h"
#include "CudaCamera.cuh"
#include "Debug.h"
#include "HittableList.h"
#include "Random.h"
#include "Material.h"
#include "Sphere.h"

__device__ glm::vec3* MaterialSoA::Albedo		  = nullptr;
__device__ float*	  MaterialSoA::Fuzz			  = nullptr;
__device__ float*	  MaterialSoA::Ior			  = nullptr;
__device__ uint32_t*  MaterialSoA::MaterialFlagsX = nullptr;
__device__ uint32_t*  MaterialSoA::MaterialFlagsY = nullptr;
__device__ uint32_t*  MaterialSoA::MaterialFlagsZ = nullptr;

#define RND (curand_uniform(&local_rand_state))

__host__ void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		// exit(99);
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

__global__ void create_world(HittableList* d_list, /*Material* d_materials,*/ BVHSoA* d_world, int nx, int ny, curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		uint16_t	i				 = 0;

		Materials::Add(MaterialType::Lambert, glm::vec3(0.5f, 0.5f, 0.5f), 0.0f, 1.0f, i);
		// Ground sphere:
		new (&d_list->m_Objects[i]) Sphere(glm::vec3(0, -1000.0f, -1), 1000.0f, i);
		i++;

		// For each grid position:
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				float	  choose_mat = RND;
				glm::vec3 center(a + RND, 0.2f, b + RND);
				if (choose_mat < 0.8f)
				{
					Materials::Add(MaterialType::Lambert, glm::vec3(RND * RND, RND * RND, RND * RND), 0.0f, 1.0f, i);
					new (&d_list->m_Objects[i]) Sphere(center, 0.2f, i);
					i++;
				}
				else if (choose_mat < 0.95f)
				{
					Materials::Add(MaterialType::Metal, glm::vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND, 1.0f, i);
					new (&d_list->m_Objects[i]) Sphere(center, 0.2f, i);
					i++;
				}
				else
				{
					Materials::Add(MaterialType::Dielectric, glm::vec3(1.0), 0.0f, 1.5f, i);
					new (&d_list->m_Objects[i]) Sphere(center, 0.2f, i);
					i++;
				}
			}
		}

		// Add the three big spheres:
		Materials::Add(MaterialType::Dielectric, glm::vec3(1.0), 0.0f, 1.5f, i);
		new (&d_list->m_Objects[i]) Sphere(glm::vec3(0, 1, 0), 1.0f, i);
		i++;
		Materials::Add(MaterialType::Lambert, glm::vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f, i);
		new (&d_list->m_Objects[i]) Sphere(glm::vec3(-4, 1, 0), 1.0f, i);
		i++;

		Materials::Add(MaterialType::Metal, glm::vec3(0.7f, 0.6f, 0.5f), 0.0f, 1.0f, i);
		new (&d_list->m_Objects[i]) Sphere(glm::vec3(4, 1, 0), 1.0f, i);
		i++;

		*rand_state = local_rand_state;

		uint32_t* indices = (uint32_t*)malloc(d_list->m_Count * sizeof(uint32_t));
		for (uint32_t index = 0; index < d_list->m_Count; ++index)
			indices[index] = index;

		d_world->root = BuildBVH_SoA(d_list, indices, 0, d_list->m_Count, d_world);
		printf("BVH created with %u nodes out of %u allocated\n", d_world->m_count, d_world->m_capacity);

		printf("BVH Root: %u\n", d_world->root);
		// DebugBVHNode(d_world, d_world->root);

		free(indices);
	}
}

//__device__ vec3 unit_vector(const vec3& v)
//{
//	float length = sqrtf(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
//	return vec3(v.x() / length, v.y() / length, v.z() / length);
//}

__device__ glm::vec3 RayColor(Ray& ray, BVHSoA* world, HittableList* list, const uint32_t depth, uint32_t& randSeed)
{
	glm::vec3 cur_attenuation(1.0f);
	Ray		  current_ray = ray;

	for (uint32_t i = 0; i < depth; i++)
	{
		HitRecord rec;
		// Use current_ray instead of ray
		if (!TraverseBVH_SoA(current_ray, 0.001f, FLT_MAX, list, world, world->root, rec))
		{
			// Sky color calculation
			glm::vec3 unit_direction = glm::normalize(current_ray.Direction());
			float	  t				 = 0.5f * (unit_direction.y + 1.0f);
			glm::vec3 sky_color		 = (1.0f - t) * glm::vec3(1.0) + t * glm::vec3(0.5f, 0.7f, 1.0f);
			return cur_attenuation * sky_color;
		}

		// Russian Roulette for path termination
		if (i > 3)
		{
			float rrProb = fmaxf(cur_attenuation.x, fmaxf(cur_attenuation.y, cur_attenuation.z));
			if (RandomFloat(randSeed) > rrProb)
				break;
			cur_attenuation /= rrProb;
		}

		// Scatter ray with optimized material interaction
		Ray		  scattered_ray;
		glm::vec3 attenuation = glm::vec3(1.0f);
		if (!Material::Scatter(current_ray, scattered_ray, rec, attenuation, randSeed, rec.MaterialIndex))
			break;

		// Update attenuation and current ray
		cur_attenuation *= attenuation;
		current_ray = scattered_ray;

		// Early termination for very low contribution
		if (fmaxf(cur_attenuation.x, fmaxf(cur_attenuation.y, cur_attenuation.z)) < 0.001f)
			break;
	}

	return glm::vec3(0.0f); // Exceeded Max depth
}

__global__ void InternalRender(glm::vec3* __restrict__ fb, BVHSoA* __restrict__ world, HittableList* list /*, Material* d_materials*/, uint32_t max_x, uint32_t max_y, Camera* camera, uint32_t samplersPerPixel, float colorMul, uint32_t maxDepth, uint32_t* randSeeds)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;

	uint32_t pixel_index = j * max_x + i;

	uint32_t  seed = randSeeds[pixel_index];
	glm::vec3 pixel_color(0.0f);
	for (uint32_t s = 0; s < samplersPerPixel; s++)
	{
		float u = float(float(i) + RandomFloat(seed)) / float(max_x);
		float v = float(float(j) + RandomFloat(seed)) / float(max_y);
		// u		 = 1.0f - u;
		v		 = 1.0f - v;
		auto ray = camera->GetRay(u, v);

		pixel_color += RayColor(ray, world, list /*, d_materials*/, maxDepth, seed);
	}
	randSeeds[pixel_index] = seed;
	pixel_color *= colorMul;
	pixel_color		= glm::sqrt(pixel_color);
	fb[pixel_index] = pixel_color;
}

__host__ void CudaRenderer::Init()
{
	cudaDeviceSetLimit(cudaLimitStackSize, 20000);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	// allocate random state
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_rand_seeds, m_Width * m_Height * sizeof(uint32_t)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	constexpr int numHitables = 22 * 22 + 1 + 3;
	CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&d_list, sizeof(HittableList)));
	CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&d_list->m_Objects, numHitables * sizeof(Sphere)));
	d_list->m_Count = numHitables;
	// CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&d_materials, numHitables * sizeof(Material)));
	// CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_world, sizeof(BVHSoA)));

	BVHSoA::Init(d_world, numHitables * 2 - 1);

	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	MaterialSoA::Init(numHitables);

	rand_init<<<1, 1>>>(d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	dim3 block(8, 8);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	render_init<<<grid, block>>>(m_Width, m_Height, d_rand_seeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	create_world<<<1, 1>>>(d_list /*, d_materials*/, d_world, m_Width, m_Height, d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	cudaDeviceSetLimit(cudaLimitStackSize, 200);
}

__host__ void CudaRenderer::Render() const
{
	dim3 block(16,16);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	float aspectRatio = float(m_Width) / float(m_Height);

	static float distance = 0.0f;
	Camera		 camera(glm::vec3(13.0f + distance, 2.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 20.0f, aspectRatio);

	CHECK_CUDA_ERRORS(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	// distance += 0.1f;

	if (distance > 10.0f)
		distance = 0.0f;

	const clock_t start = clock();
	// Render our buffer
	InternalRender<<<grid, block>>>(d_Image, d_world, d_list /*, d_materials*/, m_Width, m_Height, d_camera, m_SamplesPerPixel, m_ColorMul, m_MaxDepth, d_rand_seeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	const clock_t stop		   = clock();
	const double  timerSeconds = stop - start;
	std::cerr << "took " << timerSeconds << "ms.\n";
}

__host__ std::vector<float> CudaRenderer::CopyImage()
{
	std::vector<float> h_Pixels(m_Width * m_Height * 3 * sizeof(float));
	cudaMemcpy(h_Pixels.data(), d_Image, h_Pixels.size(), cudaMemcpyDeviceToHost);

	return h_Pixels;
}

CudaRenderer::~CudaRenderer()
{
	cudaFree(d_rand_seeds);
	cudaFree(d_rand_state2);
	cudaFree(d_list);
	// cudaFree(d_materials);
	cudaFree(d_world);
	cudaFree(d_camera);
	cudaFree(d_Image);
}
