#include "RidgidRegistration.h"
#include "cuda.h"
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
	}
}

template<unsigned int blockSize>
__global__ void reduceArray(float* arrayIn, float* arrayOut, unsigned int n)
{
	//This algorithm was used from a this website:
	// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	// accessed 4/28/2013
	
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while (i<n)
	{
		sdata[tid] += arrayIn[i] + arrayIn[i+blockSize];
		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 2048)
	{
		if (tid < 1024) 
			sdata[tid] += sdata[tid + 1024];
		__syncthreads();
	}
	if (blockSize >= 1024)
	{
		if (tid < 512) 
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (tid < 256) 
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads(); 
	}
	if (blockSize >= 128) 
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads(); 
	}

	if (tid < 32) {
		if (blockSize >= 64) 
			sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32)
			sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16)
			sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8)
			sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4)
			sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2)
			sdata[tid] += sdata[tid + 1];
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

float calcCorr( const Overlap &overlap, int deltaX, int deltaY, int deltaZ, cudaDeviceProp prop, float* deviceStaticROIimage, float* deviceStaticSum, unsigned int overlapPixelCount, float* deviceOverlapROIimage, float* deviceOverlapSum, float* staticSum, float* overlapSum, float* deviceMulImage)
{
	dim3 blocks;
	dim3 threads;
	float staticMean = 0.0f;
	float overlapMean = 0.0f;
	float staticSig = 0.0f;
	float overlapSig = 0.0f;
	float numerator = 0.0f;
	////////////////////////////////////////////////////////////
	//Find the Mean of both images
	calcBlockThread((overlap.xSize-deltaX)*(overlap.ySize-deltaY)*(overlap.zSize-deltaZ), prop, blocks, threads);

	switch (threads.x)
	{
	case 2048:
		reduceArray<2048><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<2048><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	case 1024:
		reduceArray<1024><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<1024><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	case 512:
		reduceArray<512><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<512><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	case 256:
		reduceArray<256><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<256><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	default:
		//Really? get a new card
		break;
	}

	HANDLE_ERROR(cudaMemcpy(staticSum,deviceStaticSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(overlapSum,deviceOverlapSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));

	for (int i=0; i<blocks.x; ++i)
	{
		staticMean += staticSum[i];
		overlapMean += overlapSum[i];
	}

	staticMean /= overlapPixelCount;
	overlapMean /= overlapPixelCount;
	//////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////
	//Subtract the mean off each image in place
	calcBlockThread(overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize-deltaZ, prop, blocks, threads);

	addConstantInPlace<<<blocks,threads>>>(deviceStaticROIimage,overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize-deltaZ,-staticMean);
	addConstantInPlace<<<blocks,threads>>>(deviceOverlapROIimage,overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize-deltaZ,-overlapMean);
	//////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////
	//multiply two images for the numarator
	multiplyTwoImages<<<blocks,threads>>>(deviceStaticROIimage,deviceOverlapROIimage,deviceMulImage,overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize-deltaZ);
	//////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////
	//get the standard deviation of each image
	powerInPlace<<<blocks,threads>>>(deviceStaticROIimage,overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize-deltaZ,2);
	powerInPlace<<<blocks,threads>>>(deviceOverlapROIimage,overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize-deltaZ,2);

	calcBlockThread((overlap.xSize-deltaX)*(overlap.ySize-deltaY)*(overlap.zSize-deltaZ), prop, blocks, threads);

	switch (threads.x)
	{
	case 2048:
		reduceArray<2048><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<2048><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	case 1024:
		reduceArray<1024><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<1024><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	case 512:
		reduceArray<512><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<512><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	case 256:
		reduceArray<256><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceStaticROIimage,deviceStaticSum,overlapPixelCount);
		reduceArray<256><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceOverlapROIimage,deviceOverlapSum,overlapPixelCount);
		break;
	default:
		//Really? get a new card
		break;
	}

	HANDLE_ERROR(cudaMemcpy(staticSum,deviceStaticSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(overlapSum,deviceOverlapSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));

	for (int i=0; i<blocks.x; ++i)
	{
		staticSig += staticSum[i];
		overlapSig += overlapSum[i];
	}

	staticSig /= overlapPixelCount;
	overlapSig/= overlapPixelCount;

	staticSig = sqrt(staticSig);
	overlapSig = sqrt(overlapSig);
	//////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////
	//calculate the numerator
	switch (threads.x)
	{
	case 2048:
		reduceArray<2048><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceMulImage,deviceStaticSum,overlapPixelCount);
		break;
	case 1024:
		reduceArray<1024><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceMulImage,deviceStaticSum,overlapPixelCount);
		break;
	case 512:
		reduceArray<512><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceMulImage,deviceStaticSum,overlapPixelCount);
		break;
	case 256:
		reduceArray<256><<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceMulImage,deviceStaticSum,overlapPixelCount);
		break;
	default:
		//Really? get a new card
		break;
	}

	HANDLE_ERROR(cudaMemcpy(staticSum,deviceStaticSum,sizeof(float)*blocks.x,cudaMemcpyDeviceToHost));

	for (int i=0; i<blocks.x; ++i)
		numerator += staticSum[i];
	//////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////
	//calculate the correlation
	return numerator / (staticSig*overlapSig) / overlapPixelCount;
}

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap, int& deltaXout,
	int& deltaYout, int deviceNum)
{
	cudaDeviceProp prop;
	dim3 blocks;
	dim3 threads;
	float maxCorrelation = std::numeric_limits<float>::min();
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

	HANDLE_ERROR(cudaSetDevice(deviceNum));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,deviceNum));

 	const float* roiStaticImage = staticImage->getConstFloatROIData(overlap.staticXminInd,overlap.xSize,
 		overlap.staticYminInd,overlap.ySize,overlap.staticZminInd,overlap.zSize);
 	const float* roiStaticImage2 = staticImage->getConstFloatROIData(0,1024,0,1024,overlap.staticZminInd,overlap.zSize);
 	std::string staticName1 =staticImage->getName();
 	staticName1 += "static.tif";
 
 	const float* roiOverlapImage = staticImage->getConstFloatROIData(overlap.overlapXminInd,overlap.xSize,
 		overlap.overlapYminInd,overlap.ySize,overlap.overlapZminInd,overlap.zSize);
 	std::string overlapName1 =overlapImage->getName();
 	overlapName1 += "overlap.tif";
 
 	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticImage,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticImageSmooth,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize));
 	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapImage,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapImageSmooth,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize));
 	HANDLE_ERROR(cudaMemcpy(deviceStaticImage,roiStaticImage,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize,cudaMemcpyHostToDevice));
 	HANDLE_ERROR(cudaMemcpy(deviceOverlapImage,roiOverlapImage,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize,cudaMemcpyHostToDevice));

	calcBlockThread(overlap.xSize,overlap.ySize,overlap.zSize, prop, blocks, threads);
	meanFilter<<<blocks,threads>>>(deviceStaticImage,deviceStaticImageSmooth,overlap.xSize,overlap.ySize,overlap.zSize,11);
	meanFilter<<<blocks,threads>>>(deviceOverlapImage,deviceOverlapImageSmooth,overlap.xSize,overlap.ySize,overlap.zSize,11);

	float* back = new float[overlap.xSize*overlap.ySize*overlap.zSize];
	HANDLE_ERROR(cudaMemcpy(back,deviceStaticImageSmooth,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize,cudaMemcpyDeviceToHost));
	writeImage(back,overlap.xSize,overlap.ySize,overlap.zSize,"back.tif");
	HANDLE_ERROR(cudaMemcpy(back,deviceOverlapImageSmooth,sizeof(float)*overlap.xSize*overlap.ySize*overlap.zSize,cudaMemcpyDeviceToHost));
	writeImage(back,overlap.xSize,overlap.ySize,overlap.zSize,"back2.tif");
	

	unsigned int maxOverlapPixelCount = overlap.xSize*overlap.ySize*overlap.zSize;
	calcBlockThread(maxOverlapPixelCount, prop, blocks, threads);
	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticROIimage,sizeof(float)*maxOverlapPixelCount));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapROIimage,sizeof(float)*maxOverlapPixelCount));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMulImage,sizeof(float)*maxOverlapPixelCount));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticSum,sizeof(float)*blocks.x));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapSum,sizeof(float)*blocks.x));

	float* staticSum = new float[blocks.x];
	float* overlapSum = new float[blocks.x];

	for (int deltaX=0; deltaX<MARGIN && deltaX<overlap.xSize; ++deltaX)
	{
		time_t start = time(NULL);
		for (int deltaY=0; deltaY<MARGIN && deltaY<overlap.ySize; ++deltaY)
		{
			printf(".");
			int deltaZ = 0;
// 			for (int deltaZ=0; deltaZ<MARGIN && deltaZ<overlap.zSize; ++deltaZ)
// 			{
				//get optimal blocks and threads for the image size that we have
				calcBlockThread(overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize-deltaZ, prop, blocks, threads);

				////////////////////////////////////////////////////////////
				//get the region of interest ROI of this inner for loop 
				unsigned int overlapPixelCount = (overlap.xSize-deltaX)*(overlap.ySize-deltaY)*(overlap.zSize-deltaZ);

				getROI<<<blocks,threads>>>(deviceStaticImageSmooth,overlap.xSize,overlap.ySize,overlap.zSize,deviceStaticROIimage,
					0,0,0,overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize);

				getROI<<<blocks,threads>>>(deviceOverlapImageSmooth,overlap.xSize,overlap.ySize,overlap.zSize,deviceOverlapROIimage,
					deltaX,deltaY,deltaZ,overlap.xSize-deltaX,overlap.ySize-deltaY,overlap.zSize);
				////////////////////////////////////////////////////////////

				float correlation = calcCorr(overlap, deltaX, deltaY, deltaZ, prop, deviceStaticROIimage, deviceStaticSum, overlapPixelCount, deviceOverlapROIimage, deviceOverlapSum, staticSum, overlapSum, deviceMulImage);

				if (correlation>maxCorrelation)
				{
					maxCorrelation = correlation;
					bestDeltaX = deltaX;
					bestDeltaY = deltaY;
					bestDeltaZ = deltaZ;
				}
			//}
		}
		start = time(NULL) - start;
		printf("%d:%d\n",deltaX,start);
	}

	HANDLE_ERROR(cudaFree(deviceStaticROIimage));
	HANDLE_ERROR(cudaFree(deviceOverlapROIimage));
	HANDLE_ERROR(cudaFree(deviceStaticSum));
	HANDLE_ERROR(cudaFree(deviceOverlapSum));
	HANDLE_ERROR(cudaFree(deviceMulImage));
	HANDLE_ERROR(cudaFree(deviceStaticImage));
	HANDLE_ERROR(cudaFree(deviceOverlapImage));
	delete roiStaticImage;
	delete roiOverlapImage;
	delete staticSum;
	delete overlapSum;

	printf("Delta (%d,%d,%d) max:%f\n",bestDeltaX,bestDeltaY,bestDeltaZ,maxCorrelation);
}
