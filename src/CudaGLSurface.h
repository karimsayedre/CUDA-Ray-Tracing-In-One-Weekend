#pragma once
#include <cuda_runtime_api.h>
#include <SFML/OpenGL.hpp>
#include <cuda_gl_interop.h>

struct CudaGLSurface
{
	cudaGraphicsResource* CudaRes;
	GLuint				  GlTex;
	cudaSurfaceObject_t	  SurfObj;
	cudaArray_t			  Array;
	bool				  IsMapped;

	explicit CudaGLSurface(const GLuint glTex)
		: CudaRes(nullptr), GlTex(glTex), SurfObj(0), Array(nullptr), IsMapped(false)
	{
		RegisterTexture();
	}

	~CudaGLSurface()
	{
		if (SurfObj)
			CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(SurfObj));
		UnregisterSurface();
	}

	void RegisterTexture()
	{
		CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterImage(&CudaRes, GlTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	}

	void UnregisterSurface()
	{
		if (IsMapped)
		{
			CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(1, &CudaRes, nullptr));
			IsMapped = false;
		}

		if (CudaRes)
		{
			CHECK_CUDA_ERRORS(cudaGraphicsUnregisterResource(CudaRes));
			CudaRes = nullptr;
		}
	}

	void Resize(const GLuint newTex)
	{
		// Clean up old resources
		if (SurfObj)
		{
			CHECK_CUDA_ERRORS(cudaDestroySurfaceObject(SurfObj));
			SurfObj = 0;
		}

		UnregisterSurface();
		GlTex = newTex;
		RegisterTexture();
	}

	cudaSurfaceObject_t BeginFrame()
	{
		CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &CudaRes, nullptr));
		IsMapped = true;

		CHECK_CUDA_ERRORS(cudaGraphicsSubResourceGetMappedArray(&Array, CudaRes, 0, 0));

		// Only create surface object if it doesn't exist yet
		if (SurfObj == 0)
		{
			cudaResourceDesc desc;
			desc.resType		 = cudaResourceTypeArray;
			desc.res.array.array = Array;

			CHECK_CUDA_ERRORS(cudaCreateSurfaceObject(&SurfObj, &desc));
		}

		return SurfObj;
	}

	void EndFrame()
	{
		// Don't destroy the surface object, just unmap the resource
		CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(1, &CudaRes, nullptr));
		IsMapped = false;
	}
};