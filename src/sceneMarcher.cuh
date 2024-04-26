#include <cmath>
#include "cudaMathUtils.cuh"
#include "rayIntersection.cuh"
#include "scene.cuh"

#define MAX_STEPS 200
#define MAX_REFLECTIONS 3

// #define SHOW_BOUND_HITS

struct MaterialInfo {
  float3 color;
  float roughness;
  float metalic;
};


__device__ __inline__ float sphereSDF(const float3& point, const float3& center, float radius)
{
  return distBetween3f(point, center) - radius;
}

/**
 * @param nornal normal vector of the plane (has to be normalized)
 * @param d the distance of the plane from the origin.
 */
__device__ __inline__ float planeSDF(const float3& point, const float3& normal, float d)
{
  return dot3f(point, normal) + d;
}

__device__ __inline__ float boxSDF(const float3& p, const float3& o, const float3& b)
{
  float qx = abs(p.x - o.x) - b.x;
  float qy = abs(p.y - o.y) - b.y;
  float qz = abs(p.z - o.z) - b.z;

  return norm3f(max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)) + min(max(qx, max(qy, qz)), 0.0);
}

/**
 * Signed-Distance-Function of the whole scene.
 */
template<bool sampleMaterials>
__device__ float sceneSDF(const float3& inPoint, Scene* scene, Material* outMaterial) {
  float3 point = inPoint;
  point.y += sin(inPoint.x * scene->wobbleFrequency + scene->time * scene->wobbleScroll) * scene->wobbleAmplitude;

  // Very large number to start with.
  float minDistToScene = INFINITY;

  for (int i = 0; i < scene->primitivesCount; ++i) {
    Primitive& primitive = scene->primitives[i];

    float dist;
    if (primitive.type == PrimitiveType::SPHERE)
    {
      dist = sphereSDF(point, primitive.position, primitive.data.sphere.radius);
    }
    else if (primitive.type == PrimitiveType::PLANE)
    {
      float d = -dot3f(primitive.position, primitive.data.plane.normal);
      dist = planeSDF(point, primitive.data.plane.normal, d);
    }
    else if (primitive.type == PrimitiveType::BOX)
    {
      dist = boxSDF(point, primitive.position, primitive.data.box.size);
    }

    if (dist < minDistToScene) {
      minDistToScene = dist;
      
      if (sampleMaterials) {
        *outMaterial = primitive.material;
      }
    }
  }

  return minDistToScene;
}

__device__ __forceinline__ void sampleNormals(const float3& point, Scene* scene, float3& normalOut)
{
    float epsilon = 0.001f; // arbitrary - should be smaller than any surface detail in your distance function, but not so small as to get lost in float precision
    float3 offX = make_float3(point.x + epsilon, point.y, point.z);
    float3 offY = make_float3(point.x, point.y + epsilon, point.z);
    float3 offZ = make_float3(point.x, point.y, point.z + epsilon);
    
    float centerDistance = sceneSDF<false>(point, scene, nullptr);
    float xDistance = sceneSDF<false>(offX, scene, nullptr);
    float yDistance = sceneSDF<false>(offY, scene, nullptr);
    float zDistance = sceneSDF<false>(offZ, scene, nullptr);

    normalOut.x = (xDistance - centerDistance) / epsilon;
    normalOut.y = (yDistance - centerDistance) / epsilon;
    normalOut.z = (zDistance - centerDistance) / epsilon;
}

__device__ float randFloat(float& seed) {
    float dot = seed * 12.9898;
    seed += 7867.233;

    float intPart;
    return abs(modf(sin(dot) * 43758.5453, &intPart));
}

__device__ void randomInUnitCube(float& seed, float3& out) {
    out.x = randFloat(seed) * 2 - 1;
    out.y = randFloat(seed) * 2 - 1;
    out.z = randFloat(seed) * 2 - 1;
}

__device__ void randomOnHemisphere(float& seed, const float3& normal, float3& out) {
    randomInUnitCube(seed, out);

    if (dot3f(normal, out) < 0) {
        out.x *= -1;
        out.y *= -1;
        out.z *= -1;
    }

    out.x += normal.x * 0.01f;
    out.y += normal.y * 0.01f;
    out.z += normal.z * 0.01f;
    normalize3f(out);
}

struct RayHitInfo {
  float start;
  float end;
};

__device__ uint8_t sortPrimitives(Scene* scene, const float3& rayOrigin, const float3& rayDir, RayHitInfo* hitOrder)
{
  uint8_t listLength = 0;

  float3 invRayDir = {
    1 / rayDir.x,
    1 / rayDir.y,
    1 / rayDir.z,
  };
  float3 relativeRayOrigin;

  for (int i = 0; i < scene->primitivesCount; ++i) {
    Primitive& primitive = scene->primitives[i];

    float nearHit = -1;
    float farHit = -1;
    if (primitive.type == PrimitiveType::PLANE)
    {
      if (dot3f(rayDir, primitive.data.plane.normal) < 0) {
        float d = -dot3f(primitive.position, primitive.data.plane.normal);
        nearHit = rayToPlane(rayOrigin, rayDir, primitive.data.plane.normal, d);
        farHit = INFINITY;
      }
    }
    else
    {
      float3 relativeRayOrigin = {
        rayOrigin.x - primitive.bounds.aabb.pos.x,
        rayOrigin.y - primitive.bounds.aabb.pos.y,
        rayOrigin.z - primitive.bounds.aabb.pos.z,
      };
      rayToBox(relativeRayOrigin, invRayDir, primitive.bounds.aabb.rad, nearHit, farHit);
    }

    if (nearHit < 0) {
      continue;
    }

    // Insertion sort
    RayHitInfo& el = hitOrder[listLength];
    el.start = nearHit;
    el.end = farHit;
    for (int s = listLength - 1; s >= 0; s--)
    {
      RayHitInfo& elA = hitOrder[s];
      RayHitInfo& elB = hitOrder[s + 1];
      if (elA.start <= elB.start)
      {
        // Good order
        break;
      }

      // Swap
      RayHitInfo tmp;
      tmp = elA;
      elA = elB;
      elB = tmp;
    }
    listLength++;
  }

  return listLength;
}

__device__ void multiplyWithSkyColor(const float3& dir, /*out*/ float3& outColor)
{
  static const float3 topSky = { 0.9f, 0.9f, 1.0f };
  static const float3 horizon = { 0.5f, 0.4f, 0.6f };

  outColor.x *= dir.y * topSky.x + (1 - dir.y) * horizon.x;
  outColor.y *= dir.y * topSky.y + (1 - dir.y) * horizon.y;
  outColor.z *= dir.y * topSky.z + (1 - dir.y) * horizon.z;
}

__device__ float sceneMarcher(float& seed, Scene* scene, float3 rayOrigin, float3 dir, int reflIdx, float maxDistSq, float3& outColor)
{
  static const float eps = 0.0001f;
  static const float eps2 = 0.0005f;
  static const float nudge = 0.01f;

  // Resetting color
  outColor.x = 1.0f;
  outColor.y = 1.0f;
  outColor.z = 1.0f;

  float3 normal;
  Material material;

  RayHitInfo hitOrder[MAX_PRIMITIVE_INSTANCES];

  for (int ref = 0; ref < MAX_REFLECTIONS; ++ref)
  {
    // Ray direction changed, sorting primitives again.
    uint8_t hitObjects = sortPrimitives(scene, rayOrigin, dir, /*out*/ hitOrder);

    // Did not hit any bounding boxes
    if (hitObjects == 0)
    {
      // Sky color
      multiplyWithSkyColor(dir, /*out*/ outColor);
      return;
    }

    #ifdef SHOW_BOUND_HITS
    outColor.x = min(hitObjects/3.0f, 1.0f);
    outColor.y = hitOrder[0].start / 10;
    outColor.z = 0.0f;
    return;
    #endif

    float3 pos;
    float minDist;
    float prevMinDist = -1;

    for (int b = 0; b < hitObjects; ++b)
    {
      prevMinDist = -1;

      float progress = hitOrder[b].start - eps;

      for (int i = 0; i < MAX_STEPS; ++i)
      {
        pos.x = rayOrigin.x + dir.x * progress;
        pos.y = rayOrigin.y + dir.y * progress;
        pos.z = rayOrigin.z + dir.z * progress;

        minDist = sceneSDF<false>(pos, scene, nullptr);

        // Inside volume?
        if (minDist <= 0 && prevMinDist > 0)
        {
          // No need to check more objects.
          b = hitObjects;
          break;
        }

        // Converging on surface?
        if (minDist <= 0 || (minDist < eps && minDist < prevMinDist))
        {
          // No need to check more objects.
          b = hitObjects;
          break;
        }
        
        // march forward safely
        progress += minDist;

        if (progress > hitOrder[b].end)
        {
          // Stop checking this object.
          break;
        }

        prevMinDist = minDist;
      }
    }

    // Not near surface or distance rising?
    if (minDist > eps2 || minDist > prevMinDist)
    {
      // Sky color
      multiplyWithSkyColor(dir, /*out*/ outColor);
      return;
    }

    // Sample normals
    sampleNormals(pos, scene, normal);
    // Sample material
    sceneSDF<true>(pos, scene, &material);

    if (material.emissive)
    {
      outColor.x *= material.color.x;
      outColor.y *= material.color.y;
      outColor.z *= material.color.z;
      // No further reflections necessary.
      return;
    }

    // Should some color be absorbed, or should light reflect?
    float fresnel = (1 - material.roughness) + (1 + dot3f(dir, normal)) * material.roughness;
    // Testing a more intense falloff
    fresnel = max(0., fresnel * 2 - 1);
    fresnel *= fresnel * fresnel * fresnel;

    if (randFloat(seed) < fresnel)
    {
      // Reflecting: 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
      float dn2 = 2. * dot3f(dir, normal);
      dir.x -= dn2 * normal.x;
      dir.y -= dn2 * normal.y;
      dir.z -= dn2 * normal.z;
    }
    else
    {
      // Diffuse ray
      randomOnHemisphere(seed, normal, dir);

      // Absorb color
      outColor.x *= material.color.x;
      outColor.y *= material.color.y;
      outColor.z *= material.color.z;
    }

    // preparing for next reflection
    rayOrigin = pos;
    // end of reflection
  }
}