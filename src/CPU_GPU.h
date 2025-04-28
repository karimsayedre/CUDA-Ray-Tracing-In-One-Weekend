#pragma once
#include "pch.h"

#include <random>


#define CHECK_CUDA
#ifdef CHECK_CUDA
__host__ void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line);
#define CHECK_CUDA_ERRORS(val) CheckCuda((val), #val, __FILE__, __LINE__)
#else
#define CHECK_CUDA_ERRORS(val) (val)
#endif

#define CHECK_BOOL(val)     \
	do                      \
	{                       \
		if (!(val))         \
		{                   \
			__debugbreak(); \
			assert(false);  \
		}                   \
	} while (false)


namespace sf
{
	class Image;
}

enum class ExecutionMode
{
	CPU,
	GPU
};

struct CpuRNG
{
	std::mt19937						  Engine;
	std::uniform_real_distribution<float> Dist {0.0f, 1.0f};
	CpuRNG(uint32_t seed)
		: Engine(seed)
	{
	}
	__host__ float Uniform()
	{
		return Dist(Engine);
	}
};

struct GpuRNG
{
	curandState state;
	__device__	GpuRNG(const uint64_t seed)
	{
		curand_init(seed, 0, 0, &state);
	}
	__device__ float Uniform()
	{
		return curand_uniform(&state);
	}
};

template<ExecutionMode M>
struct MemPolicy
{
	template<typename T>
	static T* Alloc(size_t n)
	{
		if constexpr (M == ExecutionMode::CPU)
		{
			return static_cast<T*>(std::malloc(n * sizeof(T)));
		}
		else if constexpr (M == ExecutionMode::GPU)
		{
			T* p = nullptr;
			CHECK_CUDA_ERRORS(cudaMalloc(&p, n * sizeof(T)));
			return p;
		}
		else
		{
			static_assert(M == ExecutionMode::CPU || M == ExecutionMode::GPU, "Unsupported ExecutionMode");
			return nullptr;
		}
	}

	static void Free(void* p)
	{
		if constexpr (M == ExecutionMode::CPU)
		{
			std::free(p);
		}
		else if constexpr (M == ExecutionMode::GPU)
		{
			CHECK_CUDA_ERRORS(cudaFree(p));
		}
		else
		{
			static_assert(M == ExecutionMode::CPU || M == ExecutionMode::GPU, "Unsupported ExecutionMode");
		}
	}
};

template<typename T>
concept IsGpuImage = std::same_as<std::remove_cvref_t<T>, cudaSurfaceObject_t>;

template<typename T>
concept IsCpuImage = std::same_as<std::remove_cvref_t<T>, sf::Image>;

template<ExecutionMode Mode, typename Image>
concept ValidImageForMode =
	(Mode == ExecutionMode::GPU && IsGpuImage<Image>) || (Mode == ExecutionMode::CPU && IsCpuImage<Image>);