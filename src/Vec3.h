#pragma once

__device__ __host__ inline Vec3 Reflect(const Vec3& v, const Vec3& n)
{
	return v - 2.0f * glm::dot(v, n) * n;
}

__device__ __host__ inline bool Refract(const Vec3& v, const Vec3& n, float niOverNt, Vec3& outRefracted)
{
	const float dt = glm::dot(v, n);
	if (float discriminant = 1.f - niOverNt * niOverNt * (1.f - dt * dt); discriminant > 0.0f)
	{
		outRefracted = niOverNt * (v - n * dt) - n * glm::sqrt(discriminant);
		return true;
	}
	return false;
}

__device__ __host__ inline float Reflectance(const float cosine, const float refIdx)
{
	// Schlick's approximation
	float r0 = (1.f - refIdx) / (1.f + refIdx);
	r0		 = r0 * r0;
	return r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
}
