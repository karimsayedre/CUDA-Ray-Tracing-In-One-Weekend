#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <thrust/sort.h>
#include "surface_functions.h"
#include <surface_indirect_functions.h>
#include <cuda_surface_types.h>
#ifdef __CUDACC__
#	if (GLM_COMPILER & GLM_COMPILER_CUDA)
#		pragma message("CUDA compiler detected")
#	else
#		error("CUDA compiler NOT detected")
#	endif
#endif

#ifdef __CUDA_ARCH__
#	define CPU_ONLY_INLINE // No need for inline, inline prevents NVCC from showing these functions in PTXAS info too.
#else
#	define CPU_ONLY_INLINE inline
#endif

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

using Vec2 = glm::vec<2, float, glm::aligned_mediump>;
using Vec3 = glm::vec<3, float, glm::aligned_mediump>;
struct __align__(16) Vec4
{
	Vec3  XYZ;
	float W;
};