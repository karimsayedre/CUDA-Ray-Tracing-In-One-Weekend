#pragma once
#include "pch.h"

#define CHECK_CUDA
#ifdef CHECK_CUDA
__host__ void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line);
#	define CHECK_CUDA_ERRORS(val) CheckCuda((val), #val, __FILE__, __LINE__)
#else
#	define CHECK_CUDA_ERRORS(val) (val)
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

template<ExecutionMode M>
struct MemPolicy
{
	template<typename T>
	[[nodiscard]] static T* Alloc(const size_t count) // count multiplied by size of primitive
	{
		if constexpr (M == ExecutionMode::CPU)
		{
			return static_cast<T*>(std::malloc(count * sizeof(T)));
		}
		else if constexpr (M == ExecutionMode::GPU)
		{
			T* p = nullptr;
			CHECK_CUDA_ERRORS(cudaMalloc(&p, count * sizeof(T)));
			return p;
		}
		else
		{
			static_assert(false, "Unsupported ExecutionMode");
			return nullptr;
		}
	}

	template<typename T>
	static void Resize(T*& oldPtr, const size_t newCount) // newCount multiplied by size of primitive
	{
		if constexpr (M == ExecutionMode::CPU)
		{
			if (oldPtr)
				std::free(oldPtr);

			oldPtr = static_cast<T*>(std::malloc(newCount * sizeof(T)));
		}
		else if constexpr (M == ExecutionMode::GPU)
		{
			if (oldPtr)
				CHECK_CUDA_ERRORS(cudaFree(oldPtr));

			CHECK_CUDA_ERRORS(cudaMalloc(&oldPtr, newCount * sizeof(T)));
		}
		else
		{
			static_assert(false, "Unsupported ExecutionMode");
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
			static_assert(false, "Unsupported ExecutionMode");
		}
	}
};

template<typename T>
concept IsGpuImage = std::same_as<std::remove_cvref_t<T>, cudaSurfaceObject_t>;

template<typename T>
concept IsCpuImage = std::same_as<std::remove_cvref_t<T>, sf::Image>;

template<ExecutionMode Mode, typename Image>
concept ValidImageForMode = (Mode == ExecutionMode::GPU && IsGpuImage<Image>) || (Mode == ExecutionMode::CPU && IsCpuImage<Image>);