#pragma once

__global__ void CreateWorld(HittableList* d_List, Materials* d_Materials, BVHSoA* d_World);
__device__ Vec3 RayColor(Ray& ray, BVHSoA* __restrict__ d_World, HittableList* __restrict__ d_List, Materials* __restrict__ d_Materials, const uint32_t depth, uint32_t& randSeed);
__global__ void InternalRender(cudaSurfaceObject_t fb, BVHSoA* __restrict__ d_World, HittableList* __restrict__ d_List, Materials* __restrict__ d_Materials, uint32_t maxX, uint32_t maxY, Camera* d_Camera, uint32_t samplersPerPixel, float colorMul, uint32_t maxDepth, uint32_t* randSeeds);
