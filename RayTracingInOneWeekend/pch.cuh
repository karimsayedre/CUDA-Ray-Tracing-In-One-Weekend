#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <sm_60_atomic_functions.h> // For Pascal and newer GPUs
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <numbers>
#include <thrust/sort.h>
#include <cuda_fp16.h>
#include "surface_functions.h"
#include <cuda_surface_types.h>
#include <math.h>

constexpr float Infinity = std::numeric_limits<float>::infinity();
constexpr float Pi		 = std::numbers::e_v<float>;

using Vec3 = glm::vec<3, float, glm::aligned_mediump>;
