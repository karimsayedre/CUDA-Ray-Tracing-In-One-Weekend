#include "pch.cuh"

#include <cuda_runtime.h>
#include <future>
#include <cuda.h>
#include "CudaRenderer.cuh"

#include <curand_kernel.h>
#include <mutex>

#include "BVH.h"
#include "CudaCamera.cuh"
#include "HittableList.h"
#include "Random.h"
#include "Material.h"
#include "Sphere.h"

#define RND (curand_uniform(&local_rand_state))

__host__ void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
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

__global__ void create_world(Sphere* d_spheres, Material* d_materials, Hittable** d_world, int nx, int ny, curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		int			i				 = 0;

		new (&d_materials[i]) Material(MaterialType::Lambert, glm::vec3(0.5f, 0.5f, 0.5f));
		// Ground sphere:
		new (&d_spheres[i]) Sphere(glm::vec3(0, -1000.0f, -1),
									 1000.0f,
									 i);
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
					new (&d_materials[i]) Material(MaterialType::Lambert, glm::vec3(RND * RND, RND * RND, RND * RND));
					// Create a Lambertian sphere
					new (&d_spheres[i]) Sphere(center,
												 0.2f,
												 i);
					i++;

				}
				else if (choose_mat < 0.95f)
				{
					new (&d_materials[i]) Material(MaterialType::Metal, glm::vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND);
					// Create a u_Metal sphere
					new (&d_spheres[i]) Sphere(center,
												 0.2f,
												 i);
					i++;

				}
				else
				{
					new (&d_materials[i]) Material(MaterialType::Dielectric, glm::vec3(1.0), 0.0f, 1.5f);
					// Create a u_Dielectric sphere
					new (&d_spheres[i]) Sphere(center,
												 0.2f,
												i);
					i++;

				}
			}
		}

		new (&d_materials[i]) Material(MaterialType::Dielectric, glm::vec3(1.0), 0.0f, 1.5f);
		// Add the three big spheres:
		new (&d_spheres[i]) Sphere(glm::vec<3, float>(0, 1, 0),
									 1.0f,
									 i);
		i++;
		new (&d_materials[i]) Material(MaterialType::Lambert, glm::vec3(0.4f, 0.2f, 0.1f));
		new (&d_spheres[i]) Sphere(glm::vec3(-4, 1, 0),
									 1.0f,
									 i);
		i++;

		new (&d_materials[i]) Material(MaterialType::Metal, glm::vec3(0.7f, 0.6f, 0.5f), 0.0f);
		new (&d_spheres[i]) Sphere(glm::vec3(4, 1, 0),
									 1.0f,
									 i);
		i++;

		*rand_state = local_rand_state;

		Hittable** spherePtrs = new Hittable*[i];
		for (int j = 0; j < i; j++)
		{
			spherePtrs[j] = reinterpret_cast<Hittable*>(&d_spheres[j]);
		}
		*d_world = new BVHNode(HittableList(spherePtrs, i), 0.0, 1.0, &local_rand_state);
		delete[] spherePtrs;
	}
}

//__device__ vec3 unit_vector(const vec3& v)
//{
//	float length = sqrtf(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
//	return vec3(v.x() / length, v.y() / length, v.z() / length);
//}

__device__ glm::vec3 RayColor(Ray& ray, Hittable* __restrict__* __restrict__ world, Material* d_materials, const uint32_t depth, uint32_t& randSeed)
{
	// Optimize by removing repeated computations and improving branching
	glm::vec3 cur_attenuation(1.0f);
	Ray		  current_ray = ray;

	for (uint32_t i = 0; i < depth; i++)
	{
		HitRecord rec;
		// Use early exit and reduce function call overhead
		if (!static_cast<BVHNode*>(*world)->Hit(current_ray, 0.001f, FLT_MAX, rec))
		{
			// Precompute sky color to reduce runtime calculations
			glm::vec3 unit_direction = glm::normalize(current_ray.Direction());
			float	  t				 = 0.5f * (unit_direction.y + 1.0f);
			glm::vec3 sky_color		 = (1.0f - t) * glm::vec3(1.0) + t * glm::vec3(0.5f, 0.7f, 1.0f);
			return cur_attenuation * sky_color;
		}

		// Russian Roulette for path termination (uncomment and modify as needed)
		//if (i > 3)
		//{
		//	float rrProb = fmaxf(cur_attenuation.x, fmaxf(cur_attenuation.y, cur_attenuation.z));
		//	if (RandomFloat(randSeed) > rrProb)
		//		break;
		//	cur_attenuation /= rrProb;
		//}

		// Scatter ray with optimized material interaction
		Ray scattered_ray;
		if (!d_materials[rec.MaterialIndex].Scatter(current_ray, scattered_ray, rec, cur_attenuation, randSeed))
			return glm::vec3(0.0f);

		current_ray = scattered_ray;

		// Early termination for very low contribution
		if (fmaxf(cur_attenuation.x, fmaxf(cur_attenuation.y, cur_attenuation.z)) < 0.001f)
			break;
	}

	return glm::vec3(0.0f); // Exceeded max depth
}

__global__ void InternalRender(glm::vec3* __restrict__ fb, Hittable* __restrict__* __restrict__ world, Material* d_materials, uint32_t max_x, uint32_t max_y, Camera* camera, uint32_t samplersPerPixel, float colorMul, uint32_t maxDepth, uint32_t* randSeeds)
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
		float u	 = float(float(i) + RandomFloat(seed)) / float(max_x);
		float v	 = float(float(j) + RandomFloat(seed)) / float(max_y);
		u		 = 1.0f - u;
		v		 = 1.0f - v;
		auto ray = camera->GetRay(u, v);

		pixel_color += RayColor(ray, world, d_materials, maxDepth, seed);
	}
	randSeeds[pixel_index] = seed;
	pixel_color *= colorMul;
	pixel_color		= glm::sqrt(pixel_color);
	fb[pixel_index] = pixel_color;
}

__host__ void CudaRenderer::Init()
{
	cudaDeviceSetLimit(cudaLimitStackSize, 4096);

	// allocate random state
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_rand_seeds, m_Width * m_Height * sizeof(uint32_t)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	constexpr int numHitables = 22 * 22 + 1 + 3;
	CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&d_list, numHitables * sizeof(Sphere)));
	CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&d_materials, numHitables * sizeof(Material)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_camera, sizeof(Camera)));

	rand_init<<<1, 1>>>(d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	dim3 block(8, 8);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	render_init<<<grid, block>>>(m_Width, m_Height, d_rand_seeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	create_world<<<1, 1>>>((Sphere*)d_list, d_materials, d_world, m_Width, m_Height, d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

__host__ void CudaRenderer::Render() const
{
	dim3 block(8, 8);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	float aspectRatio = float(m_Width) / float(m_Height);

	static float distance = 0.0f;
	Camera		 camera(glm::vec3(13.0f + distance, 2.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 20.0f, aspectRatio);

	CHECK_CUDA_ERRORS(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	distance += 0.1f;

	if (distance > 10.0f)
		distance = 0.0f;

	const clock_t start = clock();
	// Render our buffer
	InternalRender<<<grid, block>>>(d_Image, d_world, d_materials, m_Width, m_Height, d_camera, m_SamplesPerPixel, m_ColorMul, m_MaxDepth, d_rand_seeds);
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
