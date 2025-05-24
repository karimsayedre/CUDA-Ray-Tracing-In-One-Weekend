#pragma once
#include "pch.h"

[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE float Schlick(float cosTheta, float refIdx)
{
	float r0   = (1.0f - refIdx) / (1.0f + refIdx);
	r0		   = r0 * r0;
	float inv  = 1.0f - cosTheta;
	float inv5 = inv * inv * inv * inv * inv;
	return std::fmaf(1.0f - r0, inv5, r0);
}

// Branchless refract candidate: sqrtk is zero if total internal reflection
[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE Vec3 RefractBranchless(const Vec3& v, const Vec3& n, float etaiOverEtat)
{
	float dt	= dot(v, n);
	float k		= 1.0f - etaiOverEtat * etaiOverEtat * (1.0f - dt * dt);
	float sqrtk = std::sqrtf(std::fmaxf(k, 0.0f));
	return etaiOverEtat * (v - n * dt) - n * sqrtk;
}
