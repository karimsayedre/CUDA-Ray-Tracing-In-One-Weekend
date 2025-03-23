#pragma once



// Fast reciprocal square root function implementation (1/sqrt(x))
__device__ inline float custom_rsqrtf(float x)
{
#if IS_CUDA
	// On CUDA, use the built-in rsqrtf
	return rsqrtf(x);
#else
	// On CPU, implement the fast inverse square root algorithm

	// Method using union for type punning (avoids compiler warnings/errors)
	union
	{
		float f;
		int	  i;
	} conv;

	float x2 = x * 0.5f;
	conv.f	 = x;
	conv.i	 = 0x5f3759df - (conv.i >> 1);
	x		 = conv.f;

	// Newton iterations for refinement
	x = x * (1.5f - x2 * x * x); // First iteration
	// x = x * (1.5f - x2 * x * x);  // Optional second iteration for more precision

	return x;
#endif
}


__device__ inline Vec3 Reflect(const Vec3& v, const Vec3& n)
{
	return v - 2.0f * glm::dot(v, n) * n;
}

__device__ inline bool Refract(const Vec3& v, const Vec3& n, float niOverNt, Vec3& outRefracted)
{
	const float dt		   = glm::dot(v, n);
	if (float discriminant = 1.f - niOverNt * niOverNt * (1.f - dt * dt); discriminant > 0.0f)
	{
		outRefracted = niOverNt * (v - n * dt) - n * glm::sqrt(discriminant);
		return true;
	}
	return false;
}

__device__ inline float Reflectance(const float cosine, const float refIdx)
{
	// Schlick's approximation
	float r0 = (1.f - refIdx) / (1.f + refIdx);
	r0		 = r0 * r0;
	return r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
}
__device__ inline float fastLength(const Vec3& v)
{
	// Approximate length using rsqrtf
	const float lenSq = dot(v, v);
	return lenSq > 0.f ? lenSq * custom_rsqrtf(lenSq) : 0.f;
}

