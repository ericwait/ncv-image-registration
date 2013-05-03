#include "RidgidRegistration.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include "CudaUtilities.h"


__global__ void meanFilter(float* imageIn, float* imageOut, int imageWidth, int imageHeight, int imageDepth, int kernalDiameter)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x<imageWidth && y<imageHeight && z<imageDepth)
	{
		int kernalRadius = kernalDiameter/2;
		int val = 0;
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

		imageOut[x+y*imageWidth+z*imageHeight*imageWidth] = min(val/((xMax-xMin)*(yMax-yMin)*(zMax-zMin)),255);
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


void calcBlockThread(unsigned int n, cudaDeviceProp& prop, dim3& blocks, dim3& threads)
{
	if (n<prop.maxThreadsPerBlock)
	{
		threads.x = n;
		threads.y = 1;
		threads.z = 1;
		blocks.x = 1;
		blocks.y = 1;
		blocks.z = 1;
	} 
	else
	{
		threads.x = prop.maxThreadsPerBlock;
		threads.y = 1;
		threads.z = 1;
		blocks.x = ceil((float)n/prop.maxThreadsPerBlock);
		blocks.y = 1;
		blocks.z = 1;
	}
}

void calcBlockThread(unsigned int width, unsigned int height, cudaDeviceProp& prop, dim3& blocks, dim3& threads)
{
	if (width*height<prop.maxThreadsPerBlock)
	{
		threads.x = width;
		threads.y = height;
		threads.z = 1;
		blocks.x = 1;
		blocks.y = 1;
		blocks.z = 1;
	} 
	else
	{
		int dim = sqrt((float)prop.maxThreadsPerBlock);
		threads.x = dim;
		threads.y = dim;
		threads.z = 1;
		blocks.x = ceil((float)width/dim);
		blocks.y = ceil((float)height/dim);
		blocks.z = 1;
	}
}

void calcBlockThread(unsigned int width, unsigned int height, unsigned int depth, const cudaDeviceProp &prop, dim3 &blocks,
	dim3 &threads )
{
	if(width*height*depth < prop.maxThreadsPerBlock)
	{
		blocks.x = 1;
		blocks.y = 1;
		blocks.z = 1;
		threads.x = width;
		threads.y = height;
		threads.z = depth;
	}
	else
	{
		int dim = pow((float)prop.maxThreadsPerBlock,1/3.0f);
		int eqDim = dim*dim*dim;
		int extra = prop.maxThreadsPerBlock/eqDim;

		threads.x = dim * (prop.maxThreadsPerBlock/eqDim);
		threads.y = dim;
		threads.z = dim;

		blocks.x = ceil((float)width/threads.x);
		blocks.y = ceil((float)height/threads.y);
		blocks.z = ceil((float)depth/threads.z);
	}
}

//#pragma optimize("",off)
float calcCorr(int xSize, int ySize, int zSize, cudaDeviceProp prop, float* deviceStaticROIimage, float* deviceStaticSum, float* deviceOverlapROIimage, float* deviceOverlapSum, float* staticSum, float* overlapSum, float* deviceMulImage)
{
	dim3 blocks;
	dim3 threads;
	double staticMean = 0.0;
	double overlapMean = 0.0;
	double staticSig = 0.0;
	double overlapSig = 0.0;
	double numerator = 0.0;

//////////////////////////////////////////////////////////////////////////
	//Find the Mean of both images
	unsigned int overlapPixelCount = xSize*ySize*zSize;
	calcBlockThread(overlapPixelCount, prop, blocks, threads);

	blocks.x = (blocks.x+1) / 2;

	reduceArray<<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
	reduceArray<<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);

	HANDLE_ERROR(cudaMemcpy(staticSum,deviceStaticSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(overlapSum,deviceOverlapSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));

	for (int i=0; i<blocks.x; ++i)
	{
		staticMean += staticSum[i];
		overlapMean += overlapSum[i];
	}

	staticMean /= overlapPixelCount;
	overlapMean /= overlapPixelCount;
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//Subtract the mean off each image in place
	calcBlockThread(xSize,ySize,zSize, prop, blocks, threads);

	addConstantInPlace<<<blocks,threads>>>(deviceStaticROIimage,xSize,ySize,zSize,-staticMean);
	addConstantInPlace<<<blocks,threads>>>(deviceOverlapROIimage,xSize,ySize,zSize,-overlapMean);
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//multiply two images for the numarator
	multiplyTwoImages<<<blocks,threads>>>(deviceStaticROIimage,deviceOverlapROIimage,deviceMulImage,xSize,ySize,zSize);
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//get the standard deviation of each image
	powerInPlace<<<blocks,threads>>>(deviceStaticROIimage,xSize,ySize,zSize,2);
	powerInPlace<<<blocks,threads>>>(deviceOverlapROIimage,xSize,ySize,zSize,2);

	calcBlockThread(overlapPixelCount, prop, blocks, threads);
	blocks.x = (blocks.x+1) / 2;

	reduceArray<<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
	reduceArray<<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);

	HANDLE_ERROR(cudaMemcpy(staticSum,deviceStaticSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(overlapSum,deviceOverlapSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));

	for (int i=0; i<blocks.x; ++i)
	{
		staticSig += staticSum[i];
		overlapSig += overlapSum[i];
	}

	staticSig /= overlapPixelCount;
	overlapSig /= overlapPixelCount;

	staticSig = sqrt(staticSig);
	overlapSig = sqrt(overlapSig);
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//calculate the numerator
	reduceArray<<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceMulImage,deviceStaticSum,overlapPixelCount);

	HANDLE_ERROR(cudaMemcpy(staticSum,deviceStaticSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));

	for (int i=0; i<blocks.x; ++i)
		numerator += staticSum[i];
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//calculate the correlation
	return numerator / (staticSig*overlapSig) / overlapPixelCount;
}

int overlapPixels(int deltaSe, int deltaSs, int width, int x)
{
	if (deltaSe+x<0 || deltaSs+x>width)
		return 0;

	if (deltaSs+x<=0 && deltaSe+x>=width)
		return width;

	if (deltaSs+x>0 && deltaSe+x<width)
		return deltaSe-deltaSs;

	if (deltaSs+x<=0 && deltaSe+x<width)
		return deltaSe+x;

	if (deltaSs+x>0 && deltaSs+x<=width)
		return width-deltaSs-x;

	return std::numeric_limits<int>::min();
}

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap, int& deltaXout,
	int& deltaYout, int& deltaZout, int deviceNum)
{
	cudaDeviceProp prop;
	dim3 blocks;
	dim3 threads;
	float maxCorrelation = -std::numeric_limits<float>::infinity();
	int bestDeltaX = 0;
	int bestDeltaY = 0;
	int bestDeltaZ = 0;
	float* deviceStaticImage;
	float* deviceOverlapImage;
	float* deviceStaticImageSmooth;
	float* deviceOverlapImageSmooth;
	float* deviceStaticROIimage;
	float* deviceOverlapROIimage;
	float* deviceMulImage;
	float* deviceStaticSum;
	float* deviceOverlapSum;


	//TODO: check if overlap is still available for registration

	HANDLE_ERROR(cudaSetDevice(deviceNum));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,deviceNum));
 	const float* staticImageFloat = staticImage->getConstFloatROIData(0,staticImage->getWidth(),0,staticImage->getHeight(),0,staticImage->getDepth());
	unsigned int staticPixelCount = staticImage->getWidth()*staticImage->getHeight()*staticImage->getDepth();
 
 	const float* overlapImageFloat = overlapImage->getConstFloatROIData(0,overlapImage->getWidth(),0,overlapImage->getHeight(),0,overlapImage->getDepth());
	unsigned int overlapPixelCount = overlapImage->getWidth()*overlapImage->getHeight()*overlapImage->getDepth();
 	std::string overlapName1 =overlapImage->getName();
 	overlapName1 += "overlap.tif";
 
 	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticImage,sizeof(float)*staticPixelCount));
	//HANDLE_ERROR(cudaMalloc((void**)&deviceStaticImageSmooth,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize));
 	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapImage,sizeof(float)*overlapPixelCount));
	//HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapImageSmooth,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize));
 	HANDLE_ERROR(cudaMemcpy(deviceStaticImage,staticImageFloat,sizeof(float)*staticPixelCount,cudaMemcpyHostToDevice));
 	HANDLE_ERROR(cudaMemcpy(deviceOverlapImage,overlapImageFloat,sizeof(float)*overlapPixelCount,cudaMemcpyHostToDevice));
// 
// 	calcBlockThread(overlap.xSize,overlap.ySize,overlap.zSize, prop, blocks, threads);
// 	meanFilter<<<blocks,threads>>>(deviceStaticImage,deviceStaticImageSmooth,overlap.xSize,overlap.ySize,overlap.zSize,11);
// 	meanFilter<<<blocks,threads>>>(deviceOverlapImage,deviceOverlapImageSmooth,overlap.xSize,overlap.ySize,overlap.zSize,11);
	
//////////////////////////////////////////////////////////////////////////
	//Find max pixel overlap count
	int midPointX = (overlap.deltaXmax + overlap.deltaXmin) / 2.0;
	int maxOverlapPixelCountX = max(overlapPixels(overlap.deltaXse,overlap.deltaXss,staticImage->getWidth(),overlap.deltaXmin),overlapPixels(overlap.deltaXse,overlap.deltaXss,staticImage->getWidth(),overlap.deltaXmax));
	maxOverlapPixelCountX = max(maxOverlapPixelCountX,overlapPixels(overlap.deltaXse,overlap.deltaXss,staticImage->getWidth(),midPointX));

	int midPointY = (overlap.deltaYmax + overlap.deltaYmin) / 2.0;
	int maxOverlapPixelCountY = max(overlapPixels(overlap.deltaYse,overlap.deltaYss,staticImage->getHeight(),overlap.deltaYmin),overlapPixels(overlap.deltaYse,overlap.deltaYss,staticImage->getHeight(),overlap.deltaYmax));
	maxOverlapPixelCountY = max(maxOverlapPixelCountY,overlapPixels(overlap.deltaYse,overlap.deltaYss,staticImage->getHeight(),midPointY));

	int midPointZ = (overlap.deltaZmax + overlap.deltaZmin) / 2.0;
	int maxOverlapPixelCountZ = max(overlapPixels(overlap.deltaZse,overlap.deltaZss,staticImage->getDepth(),overlap.deltaZmin),overlapPixels(overlap.deltaZse,overlap.deltaZss,staticImage->getDepth(),overlap.deltaZmax));
	maxOverlapPixelCountZ = max(maxOverlapPixelCountZ,overlapPixels(overlap.deltaZse,overlap.deltaZss,staticImage->getDepth(),midPointZ));

	unsigned int maxOverlapPixelCount = maxOverlapPixelCountX*maxOverlapPixelCountY*maxOverlapPixelCountZ;
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//Set up memory space on the card for the largest possible size we need
	calcBlockThread(maxOverlapPixelCount, prop, blocks, threads);
	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticROIimage,sizeof(float)*maxOverlapPixelCount));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapROIimage,sizeof(float)*maxOverlapPixelCount));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMulImage,sizeof(float)*maxOverlapPixelCount));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticSum,sizeof(float)*(blocks.x+1)/2));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapSum,sizeof(float)*(blocks.x+1)/2));
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//Find the best correlation
	time_t mainStart, mainEnd, xStart, xEnd, yStart, yEnd, zStart, zEnd;
	double mainSec=0, xSec=0, ySec=0, zSec=0;
	float* staticSum = new float[(blocks.x+1)/2];
	float* overlapSum = new float[(blocks.x+1)/2];

	unsigned int iterations = (overlap.deltaXmax-overlap.deltaXmin)*(overlap.deltaYmax-overlap.deltaYmin)*(overlap.deltaZmax-overlap.deltaZmin);
	unsigned int curIter = 0;

	printf("Device:%d Starting on (%d,%d,%d)\n",deviceNum,overlap.deltaXmax-overlap.deltaXmin,overlap.deltaYmax-overlap.deltaYmin,overlap.deltaZmax-overlap.deltaZmin);

	time(&mainStart);
	for (int deltaX=overlap.deltaXmin; deltaX<overlap.deltaXmax; ++deltaX)
	{
		time(&xStart);
		for (int deltaY=overlap.deltaYmin; deltaY<overlap.deltaYmax; ++deltaY)
		{
			//time(&yStart);
 			for (int deltaZ=overlap.deltaZmin; deltaZ<overlap.deltaZmax; ++deltaZ)
 			{
				//time(&zStart);
				int staticXmin, staticYmin, staticZmin;
				int overlapXmin, overlapYmin, overlapZmin;
				int xSize, ySize, zSize;
				
				staticXmin = max(0,overlap.deltaXss+deltaX);
				overlapXmin = max(deltaX-overlap.deltaXss,0);
				xSize = min(staticImage->getWidth(),overlap.deltaXse+deltaX) - staticXmin;
				
				staticYmin = max(0,overlap.deltaYss+deltaY);
				overlapYmin = max(deltaY-overlap.deltaYss,0);
				ySize = min(staticImage->getHeight(),overlap.deltaYse+deltaY) - staticYmin;

				staticZmin = max(0,overlap.deltaZss+deltaZ);
				overlapZmin = max(deltaZ-overlap.deltaZss,0);
				zSize = min(staticImage->getDepth(),overlap.deltaZse+deltaZ) - staticZmin;
				
				//get optimal blocks and threads for the image size that we have
				calcBlockThread(staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth(), prop, blocks, threads);

				////////////////////////////////////////////////////////////
				//get the region of interest ROI of this inner for loop 
				unsigned int overlapPixelCount = xSize*ySize*zSize;

				getROI<<<blocks,threads>>>(deviceStaticImage,staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth(),deviceStaticROIimage,staticXmin,staticYmin,staticZmin,xSize,ySize,zSize);
				getROI<<<blocks,threads>>>(deviceOverlapImage,overlapImage->getWidth(),overlapImage->getHeight(),overlapImage->getDepth(),deviceOverlapROIimage,overlapXmin,overlapYmin,overlapZmin,xSize,ySize,zSize);
				////////////////////////////////////////////////////////////

				float correlation = calcCorr(xSize, ySize, zSize, prop, deviceStaticROIimage, deviceStaticSum, deviceOverlapROIimage, deviceOverlapSum, staticSum, overlapSum, deviceMulImage);

				if (correlation>maxCorrelation)
				{
					maxCorrelation = correlation;
					bestDeltaX = deltaX;
					bestDeltaY = deltaY;
					bestDeltaZ = deltaZ;
				}
				++curIter;
// 				time(&zEnd);
// 				zSec = difftime(zEnd,zStart);
			}
// 			time(&yEnd);
// 			ySec = difftime(yEnd,yStart);
		}
		time(&xEnd);
		xSec = difftime(xEnd,xStart);
		if (0==deltaX%20)
			printf("(%d) PerctDone:%3.2f (%d,%d,%d) Total:%4.2f avgY:%4.2f avgZ:%4.2f\n",deviceNum,(float)curIter/iterations*100,overlap.deltaXmax,overlap.deltaYmax,overlap.deltaZmax,xSec,xSec/(overlap.deltaYmax-overlap.deltaYmin),xSec/(overlap.deltaYmax-overlap.deltaYmin)*(overlap.deltaZmax-overlap.deltaZmin));
	}
	time(&mainEnd);
	mainSec = difftime(mainEnd,mainStart);
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//Clean up
	HANDLE_ERROR(cudaFree(deviceStaticROIimage));
	HANDLE_ERROR(cudaFree(deviceOverlapROIimage));
	HANDLE_ERROR(cudaFree(deviceStaticSum));
	HANDLE_ERROR(cudaFree(deviceOverlapSum));
	HANDLE_ERROR(cudaFree(deviceMulImage));
	HANDLE_ERROR(cudaFree(deviceStaticImage));
	HANDLE_ERROR(cudaFree(deviceOverlapImage));
	delete staticImageFloat;
	delete overlapImageFloat;
	delete staticSum;
	delete overlapSum;
//////////////////////////////////////////////////////////////////////////
	
	printf("Delta (%d,%d,%d) max:%f totalTime:%f avgTime:%f\n",bestDeltaX,bestDeltaY,bestDeltaZ,maxCorrelation,mainSec,mainSec/iterations);

	deltaXout = bestDeltaX;
	deltaYout = bestDeltaY;
	deltaZout = bestDeltaZ;
}
