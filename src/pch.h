#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <numbers>
#include <thrust/sort.h>
#include <cuda_fp16.h>
#include "surface_functions.h"
#include <surface_indirect_functions.h>
#include <cuda_surface_types.h>
#include <math.h>

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
#	define fmaxf			glm::max
#	define fminf			glm::min
#	define fmaf			glm::fma
#endif

using Vec2				  = glm::vec<2, float, glm::aligned_mediump>;
using Vec3				  = glm::vec<3, float, glm::aligned_mediump>;
static constexpr float PI = std::numbers::pi_v<float>;
