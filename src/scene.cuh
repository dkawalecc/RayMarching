#ifndef SCENE_H
#define SCENE_H

#include <cstdint>

#define MAX_PRIMITIVE_INSTANCES 100

#define COMPOSITE_UNION 0
#define COMPOSITE_SUBTRACT 1


struct SphereParams {
  float radius;
};

struct PlaneParams {
  float3 normal;
};

struct BoxParams {
  float3 size;
};

enum class PrimitiveType {
  SPHERE,
  PLANE,
  BOX,
};

union PrimitiveUnion {
  SphereParams sphere;
  PlaneParams plane;
  BoxParams box;
};

struct Material {
  float3 position;
  float3 color;
  bool emissive;
  float roughness; // [0 .. 1]
  float metalic; // [0 .. 1]
};

struct AxisAlignedBoxBounds {
  float3 pos;
  float3 rad;
};

struct PlaneBounds {
  float3 pos;
  float3 normal;
};

union FastBoundsUnion {
  AxisAlignedBoxBounds aabb;
  PlaneBounds plane;
};

struct Primitive {
  PrimitiveType type;
  PrimitiveUnion data;
  Material material;
  float3 position;
  FastBoundsUnion bounds;

  uint8_t compositeMode;
  float compositeBlend;
};

struct Scene {
  double time = 0.0;
  float wobbleAmplitude = 0.0;
  float wobbleFrequency = 1.0;
  float wobbleScroll = 1.0;
  
  uint32_t primitivesCount = 0;
  Primitive primitives[MAX_PRIMITIVE_INSTANCES];
};

#endif //SCENE_H