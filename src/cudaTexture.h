#ifndef CUDA_TEXTURE_H
#define CUDA_TEXTURE_H

#include "openGL.h"
#include "cuda_runtime.h"


class CudaTexture
{
    private:
        struct cudaGraphicsResource* _cudaResource;

        uint32_t _width;
        uint32_t _height;
        dim3 _dims;

    public:
        class SurfaceHandle
        {
            private:
                struct cudaGraphicsResource* _res;

            public:
                SurfaceHandle(struct cudaGraphicsResource* res);
                ~SurfaceHandle();

                cudaSurfaceObject_t surface;
        };

        CudaTexture(uint32_t width, uint32_t height);

        GLuint textureId;

        SurfaceHandle prepareSurface();
        const dim3& getDims() const;
    
};

#endif //CUDA_TEXTURE_H