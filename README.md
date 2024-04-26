# University Ray Marching Project

## Development

## Prerequisites

- Download [GLFW3 source-code](https://www.glfw.org/download.html) into a desired directory.
  - Add the directory containing CMakeLists.txt to glfw3_DIR as environment variable (e.g. if in D:\Projects\SDKs\glfw-3.3.8\CMakeLists.txt, then D:\Projects\SDKs\glfw-3.3.8).
- Download [LUA source-code](https://github.com/walterschell/Lua) into a desired directory.
  - Add the directory containing CMakeLists.txt to lua_DIR as environment variable.
- [Dev] Download and extract [OpenCV](https://opencv.org/releases/) into a deired directory.
  - Add directory containing _OpenCVConfig.cmake_ or _opencv-config.cmake_ (usually C:\opencv\build) to OpenCV_DIR as environment variable.
  - If any error occurs it might be useful to add "C:\opencv\build\x64\vc16\bin" to PATH.

### Configuring for Win32

```sh
./tools/config-win.sh
```

### Compiling

```sh
./tools/build-debug.sh
```

## References

- [Interop with OpenGL](https://forums.developer.nvidia.com/t/reading-and-writing-opengl-textures-with-cuda/31746)
- [Interop with OpenGL v2](https://www.3dgep.com/opengl-interoperability-with-cuda/)
- [No CUDA Toolkit found issue](https://github.com/NVlabs/instant-ngp/issues/18)
