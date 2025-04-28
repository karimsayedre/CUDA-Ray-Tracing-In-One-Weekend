#pragma once
#include <cuda_runtime_api.h>
#include <SFML/OpenGL.hpp>
#include <cuda_gl_interop.h>

#include "Renderer.h"


struct CudaGLSurface
{
	cudaGraphicsResource* CudaRes;
	GLuint				  GlTex; // Remember the current OpenGL texture handle

	CudaGLSurface(GLuint _glTex)
		: CudaRes(nullptr), GlTex(_glTex)
	{
		RegisterTexture();
	}

	~CudaGLSurface()
	{
		UnregisterSurface();
	}

	void RegisterTexture()
	{
		CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterImage(&CudaRes, GlTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void UnregisterSurface()
	{
		if (CudaRes)
		{
			CHECK_CUDA_ERRORS(cudaGraphicsUnregisterResource(CudaRes));
			CudaRes = nullptr;
		}
	}

	// Call this after any texture resize/recreation with a new OpenGL handle
	void Resize(const GLuint newTex)
	{
		UnregisterSurface();
		GlTex = newTex;
		RegisterTexture();
	}

	// Map the resource, create a surface object for CUDA access
	cudaSurfaceObject_t BeginFrame()
	{
		CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &CudaRes, nullptr));
		cudaArray_t array;
		CHECK_CUDA_ERRORS(cudaGraphicsSubResourceGetMappedArray(&array, CudaRes, 0, 0));

		cudaResourceDesc desc;
		desc.resType		 = cudaResourceTypeArray;
		desc.res.array.array = array;

		cudaSurfaceObject_t surf = 0;
		CHECK_CUDA_ERRORS(cudaCreateSurfaceObject(&surf, &desc));
		return surf;
	}

	// Destroy the surface and unmap the resource
	void EndFrame(const cudaSurfaceObject_t surf)
	{
		CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(surf));
		CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(1, &CudaRes, nullptr));
	}
};