#ifndef CAMERA_H
#define CAMERA_H

#include "matrix.cuh"

struct Camera {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;

    float hSpan;   // width/2
    float vSpan;   // height/2
    float near;
    float far;

    int subSamples;

    Camera(float hSpan, float vSpan, float near, float far) :
        hSpan(hSpan),
        vSpan(vSpan),
        near(near),
        far(far) {}
};

#endif CAMERA_H
