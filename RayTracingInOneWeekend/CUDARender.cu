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
	//curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);

	rand_state[pixel_index] = pcg_hash(1984 + pixel_index);
}

__global__ void create_world(Sphere* d_spheres, Hittable** d_world, int nx, int ny, curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		int			i				 = 0;

		// Ground sphere:
		new (&d_spheres[i++]) Sphere(vec3(0, -1000.0f, -1),
									 1000.0f,
									 Material(MaterialType::Lambert, vec3(0.5f, 0.5f, 0.5f)));

		// For each grid position:
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				float choose_mat = RND;
				vec3  center(a + RND, 0.2f, b + RND);
				if (choose_mat < 0.8f)
				{
					// Create a Lambertian sphere
					new (&d_spheres[i++]) Sphere(center,
												 0.2f,
												 Material(MaterialType::Lambert, vec3(RND * RND, RND * RND, RND * RND)));
				}
				else if (choose_mat < 0.95f)
				{
					// Create a u_Metal sphere
					new (&d_spheres[i++]) Sphere(center,
												 0.2f,
												 Material(MaterialType::Metal, vec3(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND));
				}
				else
				{
					// Create a u_Dielectric sphere
					new (&d_spheres[i++]) Sphere(center,
												 0.2f,
												 Material(MaterialType::Dielectric, 1.0, 0.0f, 1.5f));
				}
			}
		}

		// Add the three big spheres:
		new (&d_spheres[i++]) Sphere(vec3(0, 1, 0),
									 1.0f,
									 Material(MaterialType::Dielectric, 1.0, 0.0f, 1.5f));

		new (&d_spheres[i++]) Sphere(vec3(-4, 1, 0),
									 1.0f,
									 Material(MaterialType::Lambert, vec3(0.4f, 0.2f, 0.1f)));

		new (&d_spheres[i++]) Sphere(vec3(4, 1, 0),
									 1.0f,
									 Material(MaterialType::Metal, vec3(0.7f, 0.6f, 0.5f), 0.0f));

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

[[nodiscard]] __device__ vec3 RayColor(Ray& ray, Hittable* __restrict__* __restrict__ world, const uint32_t depth, uint32_t& randSeed)
{
	//Ray	 cur_ray		 = ray;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (uint32_t i = 0; i < depth; i++)
	{
		HitRecord rec;
		if ((BVHNode*)(*world)->Hit(ray, 0.001f, FLT_MAX, rec))
		{
			//Ray	 scattered;
			//vec3 attenuation;
			if (rec.MaterialPtr->Scatter(ray, rec, cur_attenuation, randSeed))
			{
				//cur_attenuation *= attenuation;
				//cur_ray = scattered;

				//// Russian Roulette
				// if (cur_attenuation.x() < 0.001f || cur_attenuation.y() < 0.001f || cur_attenuation.z() < 0.001f)
				//{
				//	float rrPcont = (std::max(cur_attenuation.x(), std::max(cur_attenuation.y(), cur_attenuation.z())) + 0.001f);

				//	if (curand_uniform(local_rand_state) > rrPcont)
				//		break; // Terminate the path

				//	cur_attenuation /= rrPcont; // Adjust throughput for Russian Roulette
				//}
			}
			else
			{
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else
		{
			vec3  unit_direction = unit_vector(ray.Direction());
			float t				 = 0.5f * (unit_direction.y() + 1.0f);
			vec3  c				 = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5f, 0.7f, 1.0f);
			return cur_attenuation * c;
		}
	}
	return {0.0, 0.0, 0.0}; // exceeded recursion
}

__global__ void InternalRender(vec3* __restrict__ fb, Hittable* __restrict__* __restrict__ world, uint32_t max_x, uint32_t max_y, Camera* camera, uint32_t samplersPerPixel, float colorMul, uint32_t maxDepth, uint32_t* randSeeds)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;

	uint32_t pixel_index = j * max_x + i;

	uint32_t seed = randSeeds[pixel_index];
	vec3		 pixel_color(0.0f);
	for (uint32_t s = 0; s < samplersPerPixel; s++)
	{
		float u	 = float(float(i) + RandomFloat(seed)) / float(max_x);
		float v	 = float(float(j) + RandomFloat(seed)) / float(max_y);
		u		 = 1.0f - u;
		v		 = 1.0f - v;
		auto ray = camera->GetRay(u, v);

		pixel_color += RayColor(ray, world, maxDepth, seed);
	}
	randSeeds[pixel_index] = seed;
	pixel_color *= colorMul;
	pixel_color		= vec3(sqrt(pixel_color.x()), sqrt(pixel_color.y()), sqrt(pixel_color.z()));
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
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_camera, sizeof(Camera)));

	rand_init<<<1, 1>>>(d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	dim3 block(16, 16);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	render_init<<<grid, block>>>(m_Width, m_Height, d_rand_seeds);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

	create_world<<<1, 1>>>((Sphere*)d_list, d_world, m_Width, m_Height, d_rand_state2);
	CHECK_CUDA_ERRORS(cudaGetLastError());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());


}

__host__ void CudaRenderer::Render() const
{
	dim3 block(16, 16);
	dim3 grid((m_Width + block.x - 1) / block.x,
			  (m_Height + block.y - 1) / block.y);

	float aspectRatio = float(m_Width) / float(m_Height);

	static float distance = 0.0f;
	Camera		 camera(vec3(13.0f + distance, 2.0f, 3.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), 20.0f, aspectRatio);

	CHECK_CUDA_ERRORS(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	distance += 0.1f;

	if (distance > 10.0f)
		distance = 0.0f;

	const clock_t start = clock();
	// Render our buffer
	InternalRender<<<grid, block>>>(d_Image, d_world, m_Width, m_Height, d_camera, m_SamplesPerPixel, m_ColorMul, m_MaxDepth, d_rand_seeds);
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
