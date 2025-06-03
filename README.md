# CUDA Ray Tracer: From 2.5 Seconds to 9 Milliseconds

A high-performance CUDA implementation of "Ray Tracing in One Weekend" that demonstrates the power of GPU optimization through a series of progressive improvements.

![Ray Tracer Output](assets/2560x1440_50depth_3000samples_3400ms.png)
*2560x1440 50 max depth, 3000 samples, 3400ms on an RTX 3080.*

## Performance Journey

This project started as a naive CUDA port running at **2.5 seconds per frame** and was optimized down to **9 milliseconds per frame** - a **277x speedup**! Each optimization tackled specific GPU bottlenecks:

### 🚀 Major Optimizations Implemented

| Optimization               | Improvement                   |
| -------------------------- | ----------------------------- |
| Eliminated Recursion       | **8.3x faster**               |
| Precomputed Bounding Boxes | **1.5x faster**               |
| Early Ray Termination      | **1.25x faster**              |
| Structure of Arrays (SoA)  | **2.15x faster**              |
| Memory Alignment           | **1.25x faster**              |
| Russian Roulette Sampling  | Early exit for weak rays      |
| Constant Memory Usage      | Register pressure reduction   |
| NVIDIA's RNG (PCG+LCG)     | Reduced overhead vs curand    |
| Branchless Materials       | **~15% warp efficiency gain** |
| CUDA↔OpenGL Interop        | Eliminated CPU staging        |

### 🎯 Key Technical Achievements

- **Cache Hit Rate**: Almost perfect cache hit rate, everything fit's in L1 cache
- **Memory Requests**: Reduced global memory requests 
- **Warp Efficiency**: Eliminated divergent branching in material sampling
- **Register Pressure**: Minimized through explicit stack management, constant memory, and elimination of recursion

## Features

- **CPU/GPU rendering modes**: Press F2 to toggle rendering modes 
- **BVH Acceleration Structure** with Surface Area Heuristic (SAH) construction
- **Advanced Sampling**: Russian Roulette path termination and importance sampling
- **Memory Optimized**: Structure of Arrays (SoA) layout for cache efficiency
- **Real-time Performance**: Optimized for interactive frame rates
- **Direct GPU Rendering**: CUDA↔OpenGL interop for zero-copy display
- **Frames in flight**: Less waiting for CPU each frame

## Build Instructions

### Prerequisites

- **Visual Studio 2022** (with MSVC compiler)
- **CUDA Toolkit** 12.6 or later is tested
- **CMake** 3.18 or later

### Building

1. **Clone the repository**
   ```bash
   git clone https://github.com/karimsayedre/CUDA-Ray-Tracing-In-One-Weekend.git
   cd cuda-raytracer
   ```

2. **Generate build files**
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. **Build with Visual Studio**
   ```bash
   cmake --build . 
   ```

4. **Run the ray tracer**
   ```bash
   ./Release/RayTracingInOneWeekend.exe
   ```

### CMake Configuration

The project uses a simple CMake setup that automatically detects CUDA and configures the necessary compiler flags for optimal performance.

## Technical Deep Dive

### Memory Layout Transformation

The most impactful optimization was restructuring from Object-Oriented Arrays (AoS) to Structure of Arrays (SoA):

**Before (AoS):**
```cpp
class Sphere : public Hittable {
    Vec3 center;
    float radius;
    uint32_t materialIndex;
};
```

**After (SoA):**
```cpp
struct Spheres {
    Vec3*     center;         // packed sequentially
    float*    radius;         // packed sequentially  
    uint32_t* materialIndex;  // packed sequentially
    uint32_t  count;
};
```

### BVH Traversal Optimization

Replaced recursive tree traversal with an explicit stack to eliminate register pressure:

```cpp
__device__ bool Hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const
{
    Hittable* stack[16];
    int stack_ptr = 0;
    // Iterative traversal logic...
}
```

### Custom Random Number Generation

Used a lightweight PCG+LCG from NVIDIA combination replacing CUDA's curand for significant performance gains in sampling-heavy workloads.

## Performance Profiling

Key metrics tracked throughout development:
- **Frame time** (primary metric)
- **Memory bandwidth utilization**
- **Cache hit rates** (L1/L2)
- **Register usage per thread**
- **Warp occupancy**
- **Global memory requests**

## Hardware Tested

- **NVIDIA RTX 3080** (Ampere architecture)
- Results may vary on different GPU architectures

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) - Original CPU implementation
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) - GPU optimization techniques
- [Real-Time Rendering](http://www.realtimerendering.com/) - Advanced graphics techniques
- [Nvidia Optix Advanced Samples](https://github.com/nvpro-samples/optix_advanced_samples/blob/master/src/device_include/random.h) - Fast RNG code

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- Peter Shirley for the original "Ray Tracing in One Weekend" book
- NVIDIA for the CUDA toolkit and excellent profiling tools
- The graphics programming community for optimization insights

