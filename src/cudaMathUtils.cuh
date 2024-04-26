#ifndef CUDA_MATH_UTILS
#define CUDA_MATH_UTILS

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

extern constexpr double M_PI = 3.14159265358979323846;
extern constexpr double RAD_TO_DEG = 180.0 / M_PI;
extern constexpr double DEG_TO_RAD = M_PI / 180.0;

__host__ __device__ float norm3f(float x, float y, float z)
{
    return sqrt(x*x + y*y + z*z);
}

__host__ __device__ bool equals3f(const float3& a, const float3& b)
{
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) < 0.00001f;
}

__host__ __device__ void normalize3f(float3& vec)
{
    float len = norm3f(vec.x, vec.y, vec.z);
    vec.x /= len;
    vec.y /= len;
    vec.z /= len;
}

__host__ __device__ float distBetween3f(const float3& a, const float3& b)
{
    return norm3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float dot3f(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ void cross3f(const float3& a, const float3& b, float3& c) {
    // cx = aybz - azby;
    // cy = azbx - axbz;
    // cz = axby - aybx;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
}

__host__ __device__ uchar4 make_rgba_uchar4(float r, float g, float b, float a)
{
    return make_uchar4(
        (uint8_t) (max(0., min(r, 1.)) * 255),
        (uint8_t) (max(0., min(g, 1.)) * 255),
        (uint8_t) (max(0., min(b, 1.)) * 255),
        (uint8_t) (max(0., min(a, 1.)) * 255)
    );
}

#endif //CUDA_MATH_UTILS