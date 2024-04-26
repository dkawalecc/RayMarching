#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include <cmath>
#include "openGL.h"
#include "cudaMathUtils.cuh"

constexpr double MAX_ABS_PITCH = M_PI / 2 * 0.99;
constexpr double PAN_YAW_SPEED = 0.01;
constexpr double PAN_PITCH_SPEED = 0.01;

const float MIN_DISTANCE = 1.2;
const float MAX_DISTANCE = 8;

class CameraController
{
  private:
    float2 _prevRotation;
    float2 _rotation;

    float3 _origin;
    float _distance;
    float _smoothDistance;

    bool _held = false;
    float2 _freeVel;

  public:
    CameraController() : _origin { 0, 0, 0 }, _distance(MAX_DISTANCE), _freeVel{0, 0}, _smoothDistance(MAX_DISTANCE) {}

    void startCapture()
    {
      _held = true;
    }

    void stopCapture()
    {
      _held = false;
    }

    void zoom(float delta)
    {
      _distance += delta;
      if (_distance < MIN_DISTANCE)
        _distance = MIN_DISTANCE;
      if (_distance > MAX_DISTANCE)
        _distance = MAX_DISTANCE;
    }


    void pan(double deltaX, double deltaY)
    {
      deltaX *= PAN_YAW_SPEED;
      deltaY *= PAN_PITCH_SPEED;

      _rotation.x += deltaX;
      _rotation.y -= deltaY;
    }

    void update(double deltaTime);

    void populateVectors(float3& position, float3& forward, float3& right, float3& up);

    bool isMoving()
    {
      return _held || _freeVel.x > 0.0001f || _freeVel.y > 0.0001f;
    }
};

void decayTowards(float &value, float target, float decay, float step)
{
  float diff = abs(value - target);

  if (step <= 0 || diff <= 0.0001)
  {
    value = target;
    return;
  }

  float exp = 1.0f / step;
  diff *= powf(decay, exp);

  if (value < target)
  {
    value += diff;
  }
  else
  {
    value -= diff;
  }
}

void CameraController::update(double deltaTime)
{
  if (!_held)
  {
    _rotation.x += _freeVel.x * ((float) deltaTime);
    _rotation.y += _freeVel.y * ((float) deltaTime);
    decayTowards(_freeVel.x, 0, 0.97f, deltaTime);
    decayTowards(_freeVel.y, 0, 0.97f, deltaTime);
  }
  else
  {
    _freeVel.x = (_rotation.x - _prevRotation.x) * (1.0f / (float)deltaTime);
    _freeVel.y = (_rotation.y - _prevRotation.y) * (1.0f / (float)deltaTime);
  }

  decayTowards(_smoothDistance, _distance, 0.99f, deltaTime);

  if (_rotation.y > MAX_ABS_PITCH)
  {
    _rotation.y = MAX_ABS_PITCH;
  }

  if (_rotation.y < -MAX_ABS_PITCH)
  {
    _rotation.y = -MAX_ABS_PITCH;
  }

  _prevRotation = _rotation;
}

void CameraController::populateVectors(float3& position, float3& forward, float3& right, float3& up)
{
  forward = make_float3(
    sin(_rotation.x) * cos(_rotation.y),
    sin(_rotation.y),
    cos(_rotation.x) * cos(_rotation.y)
  );
  normalize3f(forward);

  right = make_float3(
    cos(_rotation.x) * cos(_rotation.y),
    0.0,
    -sin(_rotation.x) * cos(_rotation.y)
  );
  normalize3f(right);

  cross3f(forward, right, up);

  position = _origin;
  position.x -= forward.x * _smoothDistance;
  position.y -= forward.y * _smoothDistance;
  position.z -= forward.z * _smoothDistance;
}


#endif // CAMERA_CONTROLLER_H