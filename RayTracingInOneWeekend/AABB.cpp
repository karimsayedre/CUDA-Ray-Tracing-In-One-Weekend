#include "AABB.h"

bool AABB::Hit(const Ray& r, T tMin, T tMax) const noexcept
{
	const auto lambda = [&]<int Axis>() -> bool
	{
		T invD = r.InvDirection()[Axis];
		T t0 = (m_Minimum[Axis] - r.Origin()[Axis]) * invD;
		T t1 = (m_Maximum[Axis] - r.Origin()[Axis]) * invD;
		if (invD < 0.0f)
			std::swap(t0, t1);
		tMin = t0 > tMin ? t0 : tMin;
		tMax = t1 < tMax ? t1 : tMax;
		return tMax > tMin;
	};

	if (lambda.operator()<0>())
		if (lambda.operator()<1>())
			if (lambda.operator()<2>())
				return true;

	return false;

	union { __m128 t0; float t_0[4]; };
	union { __m128 t1; float t_1[4]; };
	union { __m128 invD; float inv_D[4]; };
	invD = _mm_set_ps(1.0f, r.InvDirection().z, r.InvDirection().y, r.InvDirection().x);

	t0 = _mm_sub_ps(_mm_loadu_ps(&glm::vec4(m_Minimum, 1.0f)[0]), _mm_loadu_ps(&glm::vec4(r.Origin(), 1.0f)[0]));
	t1 = _mm_sub_ps(_mm_loadu_ps(&glm::vec4(m_Maximum, 1.0f)[0]), _mm_loadu_ps(&glm::vec4(r.Origin(), 1.0f)[0]));

	t0 = _mm_mul_ps(t0, invD);
	t1 = _mm_mul_ps(t1, invD);

	__m128 cond = _mm_cmplt_ps(invD, _mm_setzero_ps());

	SwapValues(t0, t1, cond);

	//tMin = t_0[0] > tMin ? t_0[0] : tMin;
	//tMin = t_0[1] > tMin ? t_0[1] : tMin;
	//tMin = t_0[2] > tMin ? t_0[2] : tMin;
	//
	//tMax = t_1[0] < tMax ? t_1[0] : tMax;
	//tMax = t_1[1] < tMax ? t_1[1] : tMax;
	//tMax = t_1[2] < tMax ? t_1[2] : tMax;


	__m128 tMinSSE = _mm_max_ps(t0, _mm_set1_ps(tMin));
	__m128 tMaxSSE = _mm_min_ps(t1, _mm_set1_ps(tMax));

	__m128 mask = _mm_cmpgt_ps(tMaxSSE, tMinSSE);

	return _mm_movemask_ps(mask) == 0x7;

}
