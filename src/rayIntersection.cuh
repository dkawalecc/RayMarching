#ifndef RAY_INTERSECTION_H
#define RAY_INTERSECTION_H

#include "cudaMathUtils.cuh"

/**
 * Calcs intersection and exit distances, and normal at intersection.
 * The ray must be in box/object space. If you have multiple boxes all
 * aligned to the same axis, you can precompute 1/rd. If you have
 * multiple boxes but they are not alligned to each other, use the 
 * "Generic" box intersector bellow this one.
 * 
 * @see {https://iquilezles.org/articles/boxfunctions/}
 * @author {Inigo Quilez}
 */
__device__ void rayToBox(const float3& ro, const float3& invRayDir, const float3& rad, float& nearHit, float& farHit)
{
  float nx = invRayDir.x * ro.x;
  float ny = invRayDir.y * ro.y;
  float nz = invRayDir.z * ro.z;

  float kx = abs(invRayDir.x) * rad.x;
  float ky = abs(invRayDir.y) * rad.y;
  float kz = abs(invRayDir.z) * rad.z;

  float t1x = -nx - kx;
  float t1y = -ny - ky;
  float t1z = -nz - kz;

  float t2x = -nx + kx;
  float t2y = -ny + ky;
  float t2z = -nz + kz;

  float tN = max(max(t1x, t1y), t1z);
  float tF = min(min(t2x, t2y), t2z);

  if(tN > tF || tF <0.0)
  {
    // no intersection
    nearHit = -1.0f;
    farHit = -1.0f;
  }
  else
  {
    nearHit = tN;
    farHit = tF;
  }
}

/**
 * @param pn Plane normal. Must be normalized
 */
__device__ float rayToPlane(const float3& ro, const float3& rd, const float3& pn, float d)
{
  return -(dot3f(ro, pn) + d) / dot3f(rd, pn);
}


#endif //RAY_INTERSECTION_H