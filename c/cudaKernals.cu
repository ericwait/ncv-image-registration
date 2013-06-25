#include "cudaKernals.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtilities.h"

__global__ void meanFilter(float* imageIn, float* imageOut, int imageWidth, int imageHeight, int imageDepth, int kernalDiameter)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageWidth && y<imageHeight && z<imageDepth)
	{
		int kernalRadius = kernalDiameter/2;
		float val = 0;
		int xMin = max(0,x-kernalRadius);
		int xMax = min(imageWidth,x+kernalRadius);
		int yMin = max(0,y-kernalRadius);
		int yMax = min(imageHeight,y+kernalRadius);
		int zMin = max(0,z-kernalRadius);
		int zMax = min(imageDepth,z+kernalRadius);

		for (int i=xMin; i<xMax; ++i)
		{
			for (int j=yMin; j<yMax; ++j)
			{
				for (int k=zMin; k<zMax; ++k)
					//center imageIn[x+y*imageWidth]
					val += imageIn[i+j*imageWidth+k*imageHeight*imageWidth];
			}
		}

		imageOut[x+y*imageWidth+z*imageHeight*imageWidth] = min(val/((xMax-xMin)*(yMax-yMin)*(zMax-zMin)),255.0f);
	}
}

__global__ void addConstantInPlace(float* image, int width, int height, int depth, int additive)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y	+ blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<width && y<height && z<depth)
		image[x+y*width+z*height*width] = image[x+y*width+z*height*width] + additive;
}

__global__ void powerInPlace(float* image, int width, int height, int depth, int power)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y	+ blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<width && y<height && z<depth)
		image[x+y*width+z*height*width] = pow(image[x+y*width+z*height*width],power);
}

__global__ void sqrtInPlace(float* image, int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y	+ blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<width && y<height && z<depth)
		image[x+y*width+z*height*width] = sqrt(image[x+y*width+z*height*width]);
}

__global__ void multiplyTwoImages(const float* imageIn1, const float* imageIn2, float* imageOut, int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y	+ blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<width && y<height && z<depth)
		imageOut[x+y*width+z*height*width] = imageIn1[x+y*width+z*height*width] * imageIn2[x+y*width+z*height*width];
}

__global__ void getROI(const float* imageIn, int orgSizeX, int orgSizeY, int orgSizeZ, float* imageOut, int startX, int startY, int startZ, int newWidth, int newHeight, int newDepth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y	+ blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (   x>=startX && x<orgSizeX && x<startX+newWidth
		&& y>=startY && y<orgSizeY && y<startY+newHeight
		&& z>=startZ && z<orgSizeZ && z<startZ+newDepth)
	{
		unsigned int outIndex = (x-startX)+(y-startY)*newWidth+(z-startZ)*newHeight*newWidth;
		imageOut[outIndex] = imageIn[x+y*orgSizeX+z*orgSizeY*orgSizeX];
		//imageOut[outIndex] = x;
	}
}

__global__ void reduceArray(float* arrayIn, float* arrayOut, unsigned int n)
{
	//This algorithm was used from a this website:
	// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	// accessed 4/28/2013

	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + tid;
	unsigned int gridSize = blockDim.x*2*gridDim.x;
	sdata[tid] = 0;

	while (i<n)
	{
		sdata[tid] = arrayIn[i];

		if (i+blockDim.x<n)
			sdata[tid] += arrayIn[i+blockDim.x];

		i += gridSize;
	}
	__syncthreads();


	if (blockDim.x >= 2048)
	{
		if (tid < 1024) 
			sdata[tid] += sdata[tid + 1024];
		__syncthreads();
	}
	if (blockDim.x >= 1024)
	{
		if (tid < 512) 
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if (blockDim.x >= 512)
	{
		if (tid < 256) 
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads(); 
	}
	if (blockDim.x >= 128) 
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockDim.x >= 64) 
		{
			sdata[tid] += sdata[tid + 32];
			__syncthreads(); 
		}
		if (blockDim.x >= 32)
		{
			sdata[tid] += sdata[tid + 16];
			__syncthreads(); 
		}
		if (blockDim.x >= 16)
		{
			sdata[tid] += sdata[tid + 8];
			__syncthreads(); 
		}
		if (blockDim.x >= 8)
		{
			sdata[tid] += sdata[tid + 4];
			__syncthreads(); 
		}
		if (blockDim.x >= 4)
		{
			sdata[tid] += sdata[tid + 2];
			__syncthreads(); 
		}
		if (blockDim.x >= 2)
		{
			sdata[tid] += sdata[tid + 1];
			__syncthreads(); 
		}
	}

	if (tid==0)
		arrayOut[blockIdx.x] = sdata[0];
}

__global__ void reduceImage(PixelType* imageIn, PixelType* imageOut, unsigned int inWidth, unsigned int inHeight,
	unsigned int inDepth, unsigned int outWidth, unsigned int outHeight, unsigned int outDepth, unsigned int reduction)
{
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<outWidth && y<outHeight && z<outDepth)
	{
		float val = 0;
		unsigned int xMin = x*reduction;
		unsigned int xMax = min(x*reduction+reduction,inWidth);
		unsigned int yMin = y*reduction;
		unsigned int yMax = min(y*reduction+reduction,inHeight);
		unsigned int zMin = z*(reduction/2);
		unsigned int zMax = min(z*(reduction/2)+(reduction/2),inDepth);

		for (unsigned int i=xMin; i<xMax; ++i)
		{
			for (unsigned int j=yMin; j<yMax; ++j)
			{
				for (unsigned int k=zMin; k<zMax; ++k)
					//center imageIn[x+y*imageWidth]
					val += (float)imageIn[i+j*inWidth+k*inHeight*inWidth];
			}
		}

		imageOut[x+y*outWidth+z*outHeight*outWidth] = 
			val/((xMax-xMin)*(yMax-yMin)*(zMax-zMin));
	}
}