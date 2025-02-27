//#pragma once
//#include <cuda_runtime.h>
//
//// ---------------------------------------------
//// 🔹 Basic Math Operations for float2, float3, float4
//// ---------------------------------------------
//
//// ✅ Add
//__host__ __device__ inline float2 operator+(const float2& a, const float2& b) {
//    return { a.x + b.x, a.y + b.y };
//}
//
//__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
//    return { a.x + b.x, a.y + b.y, a.z + b.z };
//}
//
//__host__ __device__ inline float4 operator+(const float4& a, const float4& b) {
//    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
//}
//
//// ✅ Subtract
//__host__ __device__ inline float2 operator-(const float2& a, const float2& b) {
//    return { a.x - b.x, a.y - b.y };
//}
//
//__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
//    return { a.x - b.x, a.y - b.y, a.z - b.z };
//}
//
//__host__ __device__ inline float4 operator-(const float4& a, const float4& b) {
//    return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
//}
//
//__host__ __device__ inline float2 operator-(float a, const float2& b) {
//    return { a - b.x, a - b.y };
//}
//
//__host__ __device__ inline float3 operator-(float a, const float3& b) {
//    return { a - b.x, a - b.y, a - b.z };
//}
//
//__host__ __device__ inline float4 operator-(float a, const float4& b) {
//    return { a - b.x, a - b.y, a - b.z, a - b.w };
//}
//
//// ✅ Multiply
//__host__ __device__ inline float2 operator*(const float2& a, float b) {
//    return { a.x * b, a.y * b };
//}
//
//__host__ __device__ inline float3 operator*(const float3& a, float b) {
//    return { a.x * b, a.y * b, a.z * b };
//}
//
//__host__ __device__ inline float4 operator*(const float4& a, float b) {
//    return { a.x * b, a.y * b, a.z * b, a.w * b };
//}
//
//__host__ __device__ inline float2 operator*(float b, const float2& a) {
//    return { a.x * b, a.y * b };
//}
//
//__host__ __device__ inline float3 operator*(float b, const float3& a) {
//    return { a.x * b, a.y * b, a.z * b };
//}
//
//__host__ __device__ inline float4 operator*(float b, const float4& a) {
//    return { a.x * b, a.y * b, a.z * b, a.w * b };
//}
//
//// ✅ Multiply (component-wise)
//__host__ __device__ inline float2 operator*(const float2& a, const float2& b) {
//    return { a.x * b.x, a.y * b.y };
//}
//
//__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
//    return { a.x * b.x, a.y * b.y, a.z * b.z };
//}
//
//__host__ __device__ inline float4 operator*(const float4& a, const float4& b) {
//    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
//}
//
//// ✅ Divide
//__host__ __device__ inline float2 operator/(const float2& a, float b) {
//    return { a.x / b, a.y / b };
//}
//
//__host__ __device__ inline float3 operator/(const float3& a, float b) {
//    return { a.x / b, a.y / b, a.z / b };
//}
//
//__host__ __device__ inline float4 operator/(const float4& a, float b) {
//    return { a.x / b, a.y / b, a.z / b, a.w / b };
//}
//
//// ---------------------------------------------
//// 🔹 Advanced Math Functions
//// ---------------------------------------------
//
////// ✅ Fused Multiply-Add (FMA) - More Accurate and Faster than (a * b + c)
////__host__ __device__ inline float2 fma(const float2& a, const float2& b, const float2& c) {
////    return { __fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y) };
////}
////
////__host__ __device__ inline float3 fma(const float3& a, const float3& b, const float3& c) {
////    return { __fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z) };
////}
////
////__host__ __device__ inline float4 fma(const float4& a, const float4& b, const float4& c) {
////    return { __fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w) };
////}
//
//// ✅ Dot Product
//__host__ __device__ inline float dot(const float2& a, const float2& b) {
//    return a.x * b.x + a.y * b.y;
//}
//
//__host__ __device__ inline float dot(const float3& a, const float3& b) {
//    return a.x * b.x + a.y * b.y + a.z * b.z;
//}
//
//__host__ __device__ inline float dot(const float4& a, const float4& b) {
//    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
//}
//
//// ✅ Cross Product (Only for float3)
//__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
//    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
//}
//
////// ✅ Normalize
////__host__ __device__ inline float3 normalize(const float3& a) {
////    float invLen = rsqrtf(dot(a, a));
////    return a * invLen;
////}
////
////__host__ __device__ inline float4 normalize(const float4& a) {
////    float invLen = rsqrtf(dot(a, a));
////    return a * invLen;
////}
//
//// ✅ Length
////__host__ __device__ inline float length(const float3& a) {
////    return sqrtf(dot(a, a));
////}
////
////__host__ __device__ inline float length(const float4& a) {
////    return sqrtf(dot(a, a));
////}
//
