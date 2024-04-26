#include "openGL.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "displayRect.h"
#include "cudaTexture.h"
#include "camera.cuh"
#include "cudaMathUtils.cuh"
#include "sceneMarcher.cuh"
#include "cameraController.cuh"
#include "fpsCounter.cuh"
#include "luaApi.cuh"

using namespace std::literals;

#define SUB_SAMPLES 16
#define FAST_SUB_SAMPLES 1

CameraController cameraCtrl;
std::unique_ptr<LuaRuntime> luaRuntime;
FPSCounter fpsCounter;

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

bool leftMouseButtonPressed = false;
bool cursorPosRecorded = false;
double prevCursorX = 0;
double prevCursorY = 0;
void cursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
  if (!cursorPosRecorded)
  {
    cursorPosRecorded = true;
  }
  else
  {
    if (leftMouseButtonPressed)
    {
      cameraCtrl.pan(xpos - prevCursorX, ypos - prevCursorY);
    }
  }

  prevCursorX = xpos;
  prevCursorY = ypos;
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  if (button == 0)
  {
    if (action == GLFW_PRESS)
    {
      leftMouseButtonPressed = true;
      cameraCtrl.startCapture();
    }

    if (action == GLFW_RELEASE)
    {
      leftMouseButtonPressed = false;
      cameraCtrl.stopCapture();
    }
  }
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    cameraCtrl.zoom(yoffset * -0.4);
}
 
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    if (key == GLFW_KEY_R && action == GLFW_PRESS)
        luaRuntime->runScene();
}

__device__ __inline__ void copySceneToShared(uint32_t tid, Scene& s_scene, Scene* scene)
{
    if (tid < scene->primitivesCount)
        s_scene.primitives[tid] = scene->primitives[tid];

    if (tid == 0)
    {
        s_scene.time = scene->time;
        s_scene.wobbleAmplitude = scene->wobbleAmplitude;
        s_scene.wobbleFrequency = scene->wobbleFrequency;
        s_scene.wobbleScroll = scene->wobbleScroll;

        s_scene.primitivesCount = scene->primitivesCount;
    }
}

__device__ __inline__ void prepareRay(uint32_t x, uint32_t y, const dim3& dims, Camera* camera, float3& pos, float3& dir)
{
    float viewportX = ((float) x) / dims.x * 2. - 1.;
    float viewportY = ((float) y) / dims.y * 2. - 1;

    // The camera's position
    pos = camera->position;
    dir = camera->forward;
    dir.x += camera->right.x * viewportX * camera->hSpan + camera->up.x * viewportY * camera->vSpan;
    dir.y += camera->right.y * viewportX * camera->hSpan + camera->up.y * viewportY * camera->vSpan;
    dir.z += camera->right.z * viewportX * camera->hSpan + camera->up.z * viewportY * camera->vSpan;
    normalize3f(dir);
}

__device__ void renderSample(uint32_t x, uint32_t y, uint32_t z, dim3 dims, Camera* camera, Scene* scene, float time, float3& outColor)
{
    float3 rayPos, rayDir;
    prepareRay(x, y, dims, camera, rayPos, rayDir);

    float seed = x + y * 5857.7867 + time * 5503.5479 + z * 443.1303;

    outColor.x = 0.0f;
    outColor.y = 0.0f;
    outColor.z = 0.0f;

    float3 partialColor = make_float3(0, 0, 0);
    for (int i = 0; i < camera->subSamples; ++i) {
        float3 subRayPos = rayPos;
        subRayPos.x += randFloat(seed) * 0.005 - 0.0025;
        subRayPos.y += randFloat(seed) * 0.005 - 0.0025;
        sceneMarcher(seed, scene, subRayPos, rayDir, 0, camera->far, partialColor);
        outColor.x += partialColor.x / camera->subSamples;
        outColor.y += partialColor.y / camera->subSamples;
        outColor.z += partialColor.z / camera->subSamples;
    }
}

template<uint8_t blockSize, uint8_t subSamples>
__global__ void sceneToSurfaceKernel(cudaSurfaceObject_t surface, dim3 texDim, Camera* camera, Scene* scene, float time)
{
    __shared__ Scene s_scene;
    __shared__ float3 s_samplesBuffer[blockSize * blockSize * subSamples];
    
    // Copying to shared memory
    uint32_t zSize = blockSize * blockSize;
    uint8_t tid = threadIdx.z * zSize + threadIdx.y * blockSize + threadIdx.x;
    copySceneToShared(tid, s_scene, scene);

    __syncthreads();

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= texDim.x || y >= texDim.y)
    {
        return;
    }

    renderSample(x, y, threadIdx.z, texDim, camera, &s_scene, time, s_samplesBuffer[tid]);

    __syncthreads();

    if (threadIdx.z != 0) {
        return;
    }    

    float3 acc = { 0.0f, 0.0f, 0.0f };
    acc.x += s_samplesBuffer[tid + zSize*0].x; acc.y += s_samplesBuffer[tid + zSize*0].y; acc.z += s_samplesBuffer[tid + zSize*0].z;
    if (subSamples >= 1) {acc.x += s_samplesBuffer[tid + zSize*1].x; acc.y += s_samplesBuffer[tid + zSize*1].y; acc.z += s_samplesBuffer[tid + zSize*1].z;}
    if (subSamples >= 2) {acc.x += s_samplesBuffer[tid + zSize*2].x; acc.y += s_samplesBuffer[tid + zSize*2].y; acc.z += s_samplesBuffer[tid + zSize*2].z;}
    if (subSamples >= 3) {acc.x += s_samplesBuffer[tid + zSize*3].x; acc.y += s_samplesBuffer[tid + zSize*3].y; acc.z += s_samplesBuffer[tid + zSize*3].z;}

    uchar4 pixel = make_rgba_uchar4(acc.x / subSamples, acc.y / subSamples, acc.z / subSamples, 1.);

    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

template<uint8_t blockSize, uint8_t subSamples>
__global__ void sceneToBufferKernel(uchar4* output, dim3 texDim, Camera* camera, Scene* scene, float time)
{
    __shared__ Scene s_scene;
    __shared__ float3 s_samplesBuffer[blockSize * blockSize * subSamples];
    
    // Copying to shared memory
    uint32_t zSize = blockSize * blockSize;
    uint8_t tid = threadIdx.z * zSize + threadIdx.y * blockSize + threadIdx.x;
    copySceneToShared(tid, s_scene, scene);

    __syncthreads();

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= texDim.x || y >= texDim.y)
    {
        return;
    }

    renderSample(x, y, threadIdx.z, texDim, camera, &s_scene, time, s_samplesBuffer[tid]);

    __syncthreads();

    if (threadIdx.z != 0) {
        return;
    }    

    float3 acc = { 0.0f, 0.0f, 0.0f };
    acc.x += s_samplesBuffer[tid + zSize*0].x; acc.y += s_samplesBuffer[tid + zSize*0].y; acc.z += s_samplesBuffer[tid + zSize*0].z;
    if (subSamples >= 1) {acc.x += s_samplesBuffer[tid + zSize*1].x; acc.y += s_samplesBuffer[tid + zSize*1].y; acc.z += s_samplesBuffer[tid + zSize*1].z;}
    if (subSamples >= 2) {acc.x += s_samplesBuffer[tid + zSize*2].x; acc.y += s_samplesBuffer[tid + zSize*2].y; acc.z += s_samplesBuffer[tid + zSize*2].z;}
    if (subSamples >= 3) {acc.x += s_samplesBuffer[tid + zSize*3].x; acc.y += s_samplesBuffer[tid + zSize*3].y; acc.z += s_samplesBuffer[tid + zSize*3].z;}
    if (subSamples >= 4) {acc.x += s_samplesBuffer[tid + zSize*4].x; acc.y += s_samplesBuffer[tid + zSize*4].y; acc.z += s_samplesBuffer[tid + zSize*4].z;}
    if (subSamples >= 5) {acc.x += s_samplesBuffer[tid + zSize*5].x; acc.y += s_samplesBuffer[tid + zSize*5].y; acc.z += s_samplesBuffer[tid + zSize*5].z;}
    if (subSamples >= 6) {acc.x += s_samplesBuffer[tid + zSize*6].x; acc.y += s_samplesBuffer[tid + zSize*6].y; acc.z += s_samplesBuffer[tid + zSize*6].z;}
    if (subSamples >= 7) {acc.x += s_samplesBuffer[tid + zSize*7].x; acc.y += s_samplesBuffer[tid + zSize*7].y; acc.z += s_samplesBuffer[tid + zSize*7].z;}

    uchar4 pixel = make_rgba_uchar4(acc.x / subSamples, acc.y / subSamples, acc.z / subSamples, 1.);

    output[y * texDim.x + x] = pixel;
}

void renderTexture(CudaTexture& texture, Camera* d_camera, Scene* d_scene, double time)
{
    auto handle = texture.prepareSurface();

    const auto& texDim = texture.getDims();
    // 256 threads per block in total
    dim3 thread(8, 8, 4);
    dim3 block(texDim.x / thread.x, texDim.y / thread.y);
    sceneToSurfaceKernel<8, 4><<< block, thread >>>(handle.surface, texDim, d_camera, d_scene, time);

    // Surface is cleared with destructor.
}

void renderImageBuffer(uchar4* outputData, dim3 texDim, Camera* d_camera, Scene* d_scene, double time)
{
    // 256 threads per block in total
    dim3 thread(8, 8, 4);
    dim3 block(texDim.x / thread.x, texDim.y / thread.y);
    sceneToBufferKernel<8, 4><<< block, thread >>>(outputData, texDim, d_camera, d_scene, time);
}

void printCUDADevicesInfo()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

/**
 * Show memory usage of the GPU
 */
void printCUDAMemoryUsage()
{
    size_t free_byte;
    size_t total_byte;

    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

void windowMain(int w, int h, Camera& camera, Camera* d_camera)
{
    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);
 
    if (!glfwInit())
        exit(EXIT_FAILURE);
 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
 
    window = glfwCreateWindow(1024, 1024, "Ray Marching", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
 
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);
 
    // NOTE: OpenGL error checks have been omitted for brevity
 
    DisplayRect displayRect;
    CudaTexture outputTexture(w, h);

    printCUDADevicesInfo();

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        double now = glfwGetTime();
        double deltaTime = now - lastTime;
        lastTime = now;

        fpsCounter.registerFrame(deltaTime);

        float ratio;
        int width, height;
        // mat4x4 m, p, mvp;
 
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
 
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
 
        cameraCtrl.update(deltaTime);
        cameraCtrl.populateVectors(camera.position, camera.forward, camera.right, camera.up);
        camera.subSamples = cameraCtrl.isMoving() ? FAST_SUB_SAMPLES : SUB_SAMPLES;
	    cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

        luaRuntime->setTime(glfwGetTime());
        luaRuntime->callUpdate(deltaTime);
        renderTexture(outputTexture, d_camera, luaRuntime->d_scene, glfwGetTime());

        displayRect.render(outputTexture.textureId);

        // printCUDAMemoryUsage();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

 
    glfwDestroyWindow(window);
 
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
    bool saveImage = false;
    if(argc <= 1) 
    {
        printf("Usage: ./UniRayMarching.exe [scene] [interactive/image]");
        return EXIT_FAILURE;
    }

    if (argc > 2 && argv[2] == "image"s) {
        saveImage = true;
    }

    luaRuntime = std::make_unique<LuaRuntime>(argv[1]);
    luaRuntime->runScene();

    Camera* d_camera = nullptr;
    Camera camera(1, 1, 1, 1000);
    camera.subSamples = SUB_SAMPLES;
    cudaMalloc(&d_camera, sizeof(Camera));
    cameraCtrl.update(0);
    cameraCtrl.populateVectors(camera.position, camera.forward, camera.right, camera.up);
	cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

    int w = 512, h = 512;

    if (saveImage) {
        uchar4* deviceData;
        cudaMalloc(&deviceData, w * h * sizeof(uchar4));

        renderImageBuffer(deviceData, dim3(w, h), d_camera, luaRuntime->d_scene, glfwGetTime());

        std::vector<uchar4> hostData(w * h);
        cudaMemcpy(hostData.data(), deviceData, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost);

        cv::Mat outputImage(h, w, CV_8UC4, hostData.data());
        cv::flip(outputImage, outputImage, 0);
        cv::cvtColor(outputImage, outputImage, cv::COLOR_RGB2BGR);
        cv::imwrite("output_image.jpg", outputImage);

        cudaFree(deviceData);
    }
    else {
        windowMain(w, h, camera, d_camera);
    }

    cudaFree(d_camera);
    luaRuntime.reset();
}
 