#include "cudaTexture.h"
#include "cuda_gl_interop.h"

CudaTexture::CudaTexture(uint32_t width, uint32_t height) : _width(width), _height(height), _dims(width, height)
{
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    auto e = cudaGraphicsGLRegisterImage(&_cudaResource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

CudaTexture::SurfaceHandle CudaTexture::prepareSurface()
{
    return CudaTexture::SurfaceHandle(_cudaResource);
}

const dim3& CudaTexture::getDims() const
{
    return _dims;
}


CudaTexture::SurfaceHandle::SurfaceHandle(struct cudaGraphicsResource* res) : _res(res)
{
    auto e = cudaGraphicsMapResources(1, &res);
    cudaArray_t writeArray;
    e =  cudaGraphicsSubResourceGetMappedArray(&writeArray, res, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    e = cudaCreateSurfaceObject(&surface, &wdsc);
}

CudaTexture::SurfaceHandle::~SurfaceHandle()
{
    auto e = cudaDestroySurfaceObject(surface);
    e = cudaGraphicsUnmapResources(1, &_res);
    e = cudaStreamSynchronize(0);
}
