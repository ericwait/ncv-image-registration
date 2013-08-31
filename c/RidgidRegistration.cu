/*#include "RidgidRegistration.h"
#include "cudaKernals.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include "CudaUtilities.h"
#include "Vec.h"

unsigned int gMaxOverlapPixels = 0;


struct  corrReport
{
	Vec<int> delta;
	double correlation;
	double staticSig;
	double overlapSig;
	unsigned int nVoxels;
};

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
		int dim = (int)pow((float)prop.maxThreadsPerBlock,1/3.0f);
		int extra = (prop.maxThreadsPerBlock-dim*dim*dim)/(dim*dim);
		threads.x = dim + extra;
		threads.y = dim;
		threads.z = dim;

		blocks.x = (unsigned int)ceil((float)width/threads.x);
		blocks.y = (unsigned int)ceil((float)height/threads.y);
		blocks.z = (unsigned int)ceil((float)depth/threads.z);
	}
}

double calcCorr(int xSize, int ySize, int zSize, cudaDeviceProp prop, float* deviceStaticROIimage, float* deviceStaticSum,
	float* deviceOverlapROIimage, float* deviceOverlapSum, float* staticSum, float* overlapSum, float* deviceMulImage,
	double& staticSig, double& overlapSig)
{
#ifdef _DEBUG
	assert(xSize*ySize*zSize<=gMaxOverlapPixels);
#endif // _DEBUG

	dim3 blocks;
	dim3 threads;
	double staticMean = 0.0;
	double overlapMean = 0.0;
	staticSig = 0.0;
	overlapSig = 0.0;
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
	
// 	if (staticMean<2 || overlapMean<2)
// 		return -5;
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//Subtract the mean off each image in place
	calcBlockThread(xSize,ySize,zSize, prop, blocks, threads);

	addConstantInPlace<<<blocks,threads>>>(deviceStaticROIimage,xSize,ySize,zSize,-staticMean);
	addConstantInPlace<<<blocks,threads>>>(deviceOverlapROIimage,xSize,ySize,zSize,-overlapMean);
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//multiply two images for the numerator
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

//  	if (staticSig<1 || overlapSig<1)
//  		return -5.0;
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

	if (deltaSs+x<=0 && deltaSe+x>=(width-1))
		return width;

	if (deltaSs+x>=0 && deltaSe+x<=(width-1))
		return deltaSe-deltaSs+1;

	if (deltaSs+x<=0 && deltaSe+x<width)
		return deltaSe+x+1;

	if (deltaSs+x>0)
		return width-(deltaSs+x);

	return std::numeric_limits<int>::min();
}

void calcMaxROIs(const Overlap& overlap, Vec<int> imageExtents, comparedImages<int>& imStarts, comparedImages<int>& imSizes, Vec<int>& maxOverlapSize, cudaDeviceProp prop)
{
	imStarts.staticIm.x = std::numeric_limits<int>::max();
	imStarts.staticIm.y = std::numeric_limits<int>::max();
	imStarts.staticIm.z = std::numeric_limits<int>::max();
	imStarts.overlapIm.x = std::numeric_limits<int>::max();
	imStarts.overlapIm.y = std::numeric_limits<int>::max();
	imStarts.overlapIm.z = std::numeric_limits<int>::max();

	imSizes.staticIm.x = 0;
	imSizes.staticIm.y = 0;
	imSizes.staticIm.z = 0;
	imSizes.overlapIm.x = 0;
	imSizes.overlapIm.y = 0;
	imSizes.overlapIm.z = 0;

	for (int deltaX=overlap.deltaXmin; deltaX<overlap.deltaXmax; ++deltaX)
	{
		for (int deltaY=overlap.deltaYmin; deltaY<overlap.deltaYmax; ++deltaY)
		{
			for (int deltaZ=overlap.deltaZmin; deltaZ<overlap.deltaZmax; ++deltaZ)
			{
				comparedImages<int> mins;
				Vec<int> szs;

				mins.staticIm.x = max(0,overlap.deltaXss+deltaX);
				mins.overlapIm.x = max(0,-(overlap.deltaXss+deltaX));
				szs.x = min(imageExtents.x,overlap.deltaXse+deltaX+1) - mins.staticIm.x;

				mins.staticIm.y = max(0,overlap.deltaYss+deltaY);
				mins.overlapIm.y = max(0,-(overlap.deltaYss+deltaY));
				szs.y = min(imageExtents.x,overlap.deltaYse+deltaY+1) - mins.staticIm.y;

				mins.staticIm.z = max(0,overlap.deltaZss+deltaZ);
				mins.overlapIm.z = max(0,-(overlap.deltaZss+deltaZ));
				szs.z = min(imageExtents.z,overlap.deltaZse+deltaZ+1) - mins.staticIm.z;

				imStarts.staticIm.x = min(imStarts.staticIm.x,mins.staticIm.x);
				imStarts.staticIm.y = min(imStarts.staticIm.y,mins.staticIm.y);
				imStarts.staticIm.z = min(imStarts.staticIm.z,mins.staticIm.z);

				imSizes.staticIm.x = max(imSizes.staticIm.x,mins.staticIm.x+szs.x);
				imSizes.staticIm.y = max(imSizes.staticIm.y,mins.staticIm.y+szs.y);
				imSizes.staticIm.z = max(imSizes.staticIm.z,mins.staticIm.z+szs.z);

				imStarts.overlapIm.x = min(imStarts.overlapIm.x,mins.overlapIm.x);
				imStarts.overlapIm.y = min(imStarts.overlapIm.y,mins.overlapIm.y);
				imStarts.overlapIm.z = min(imStarts.overlapIm.z,mins.overlapIm.z);

				imSizes.overlapIm.x = max(imSizes.overlapIm.x,mins.overlapIm.x+szs.x);
				imSizes.overlapIm.y = max(imSizes.overlapIm.y,mins.overlapIm.y+szs.y);
				imSizes.overlapIm.z = max(imSizes.overlapIm.z,mins.overlapIm.z+szs.z);

				maxOverlapSize.x = max(maxOverlapSize.x,szs.x);
				maxOverlapSize.y = max(maxOverlapSize.y,szs.y);
				maxOverlapSize.z = max(maxOverlapSize.z,szs.z);
			}
		}
	}

	imSizes.staticIm.x -= imStarts.staticIm.x;
	imSizes.staticIm.y -= imStarts.staticIm.y;
	imSizes.staticIm.z -= imStarts.staticIm.z;
	imSizes.overlapIm.x -= imStarts.overlapIm.x;
	imSizes.overlapIm.y -= imStarts.overlapIm.y;
	imSizes.overlapIm.z -= imStarts.overlapIm.z;
}

//#pragma optimize("",off)
void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap,
	Vec<int>& bestDelta, double& maxCorrOut, unsigned int& bestN, int deviceNum, const char* fileName)
{
	cudaDeviceProp prop;
	dim3 blocks;
	dim3 threads;
	double maxCorrelation = -std::numeric_limits<double>::infinity();
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

	comparedImages<int> imStarts, imSizes;
	Vec<int> maxOverlapSize;
	Vec<int> staticImageExtents(staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth());

	calcMaxROIs(overlap,staticImageExtents,imStarts,imSizes,maxOverlapSize,prop);

	//check that we are within the memory space of the card
	gMaxOverlapPixels = maxOverlapSize.x*maxOverlapSize.y*maxOverlapSize.z;
	unsigned int staticPixelCount = imSizes.staticIm.x*imSizes.staticIm.y*imSizes.staticIm.z;
	unsigned int overlapPixelCount = imSizes.overlapIm.x*imSizes.overlapIm.y*imSizes.overlapIm.z;

	size_t usableMem = (deviceNum==0 ? prop.totalGlobalMem*.6 : prop.totalGlobalMem*.8);
	if (usableMem < (gMaxOverlapPixels*4 + staticPixelCount + overlapPixelCount)*sizeof(float))
	{
		bestDelta.x = 0;
		bestDelta.y = 0;
		bestDelta.z = 0;
		bestN = 1;
		maxCorrelation = -std::numeric_limits<double>::infinity();

		printf("(%d) OUT OF MEMORY FOR THIS OVERLAP!!!! Need:%d\n",deviceNum,(gMaxOverlapPixels*4 + staticPixelCount + overlapPixelCount)*sizeof(float)-prop.totalGlobalMem);
		return;
	}

 	const float* staticImageFloat = staticImage->getConstFloatROIData(imStarts.staticIm.x,imSizes.staticIm.x,imStarts.staticIm.y,imSizes.staticIm.y,imStarts.staticIm.z,imSizes.staticIm.z); 
	const float* overlapImageFloat = overlapImage->getConstFloatROIData(imStarts.overlapIm.x,imSizes.overlapIm.x,imStarts.overlapIm.y,imSizes.overlapIm.y,imStarts.overlapIm.z,imSizes.overlapIm.z);
 
 	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticImage,sizeof(float)*staticPixelCount));
	//HANDLE_ERROR(cudaMalloc((void**)&deviceStaticImageSmooth,sizeof(float)*staticPixelCount));
 	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapImage,sizeof(float)*overlapPixelCount));
	//HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapImageSmooth,sizeof(float)*overlapPixelCount));
 	HANDLE_ERROR(cudaMemcpy(deviceStaticImage,staticImageFloat,sizeof(float)*staticPixelCount,cudaMemcpyHostToDevice));
 	HANDLE_ERROR(cudaMemcpy(deviceOverlapImage,overlapImageFloat,sizeof(float)*overlapPixelCount,cudaMemcpyHostToDevice));
//////////////////////////////////////////////////////////////////////////
	//Set up memory space on the card for the largest possible size we need
	calcBlockThread(gMaxOverlapPixels, prop, blocks, threads);
	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticROIimage,sizeof(float)*gMaxOverlapPixels));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapROIimage,sizeof(float)*gMaxOverlapPixels));
	HANDLE_ERROR(cudaMalloc((void**)&deviceMulImage,sizeof(float)*gMaxOverlapPixels));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStaticSum,sizeof(float)*(blocks.x+1)/2));
	HANDLE_ERROR(cudaMalloc((void**)&deviceOverlapSum,sizeof(float)*(blocks.x+1)/2));
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
	//Find the best correlation

	time_t mainStart, mainEnd, smoothStart, smoothEnd, xStart, xEnd, yStart, yEnd, zStart, zEnd;
	double mainSec=0, smoothSec=0, xSec=0, ySec=0, zSec=0;
	float* staticSum = new float[(blocks.x+1)/2];
	float* overlapSum = new float[(blocks.x+1)/2];
	Vec<int> deltaMins(overlap.deltaXmin,overlap.deltaYmin,overlap.deltaZmin);
	Vec<int> deltaMaxs(overlap.deltaXmax,overlap.deltaYmax,overlap.deltaZmax);

	time(&mainStart);
	int smoothness = 1;
// 	for (int smoothness=27; smoothness>=1; smoothness/=3)
 	{
		Vec<int> blockMaxSizes;

		blockMaxSizes.x = max(imSizes.staticIm.x,imSizes.overlapIm.x);
		blockMaxSizes.y = max(imSizes.staticIm.y,imSizes.overlapIm.y);
		blockMaxSizes.z = max(imSizes.staticIm.z,imSizes.overlapIm.z);

		calcBlockThread(blockMaxSizes.x, blockMaxSizes.y, blockMaxSizes.z, prop, blocks, threads);

		if (smoothness>100)
		{
			meanFilter<<<blocks,threads>>>(deviceStaticImage,deviceStaticImageSmooth,
				imSizes.staticIm.x,imSizes.staticIm.y,imSizes.staticIm.z,9);

			meanFilter<<<blocks,threads>>>(deviceOverlapImage,deviceOverlapImageSmooth,
				imSizes.overlapIm.x,imSizes.overlapIm.y,imSizes.overlapIm.z,9);
		}

		unsigned int iterations = max(1,(deltaMaxs.x-deltaMins.x)*(deltaMaxs.y-deltaMins.y)*(deltaMaxs.z-deltaMins.z)/smoothness);
		unsigned int curIter = 0;
		Vec<int> deltaSizes((deltaMaxs.x-deltaMins.x),(deltaMaxs.y-deltaMins.y),(deltaMaxs.z-deltaMins.z));

		corrReport* report = new corrReport[deltaSizes.product()];
		Vec<int> reportInd(0,0,0);

		time(&smoothStart);
		for (int deltaX=deltaMins.x; deltaX<deltaMaxs.x; deltaX+=smoothness, ++reportInd.x)
		{
			reportInd.y = 0;
			time(&xStart);
			for (int deltaY=deltaMins.y; deltaY<deltaMaxs.y; deltaY+=smoothness, ++reportInd.y)
			{
				reportInd.z = 0;
				//time(&yStart);
				for (int deltaZ=deltaMins.z; deltaZ<deltaMaxs.z; deltaZ+=smoothness, ++reportInd.z)
				{
					//time(&zStart);

					// 	int deltaX = 0;
					// 	int deltaY = 0;
					// 	int deltaZ = 0;
					comparedImages<int> starts;
					Vec<int> szs;

					starts.staticIm.x = max(0,overlap.deltaXss+deltaX-imStarts.staticIm.x);
					starts.overlapIm.x = max(0,-(overlap.deltaXss+deltaX)-imStarts.overlapIm.x);
					szs.x = min(imSizes.staticIm.x,overlap.deltaXse+deltaX+1) - starts.staticIm.x;

					starts.staticIm.y = max(0,overlap.deltaYss+deltaY-imStarts.staticIm.y);
					starts.overlapIm.y = max(0,-(overlap.deltaYss+deltaY)-imStarts.overlapIm.y);
					szs.y = min(imSizes.staticIm.y,overlap.deltaYse+deltaY+1) - starts.staticIm.y;

					starts.staticIm.z = max(0,overlap.deltaZss+deltaZ-imStarts.staticIm.z);
					starts.overlapIm.z = max(0,-(overlap.deltaZss+deltaZ)-imStarts.overlapIm.z);
					szs.z = min(imSizes.staticIm.z,overlap.deltaZse+deltaZ+1) - starts.staticIm.z;

#ifdef _DEBUG
					if (szs.x>maxOverlapSize.x || szs.y>maxOverlapSize.y|| szs.z>maxOverlapSize.z)
					{
						printf("This sub image is too big!!\n");
					}
#endif // _DEBUG

					//get optimal blocks and threads for the image size that we have
					//calcBlockThread(blockMaxSizes.x, blockMaxSizes.y, blockMaxSizes.z, prop, blocks, threads);

					////////////////////////////////////////////////////////////
					//get the region of interest ROI of this inner for loop 
					//unsigned int overlapPixelCount = szs.x*szs.y*szs.z;

// 					if (smoothness>100)
// 					{
// 						getROI<<<blocks,threads>>>(deviceStaticImageSmooth,imSizes.staticIm.x,imSizes.staticIm.y,imSizes.staticIm.z,
// 							deviceStaticROIimage,starts.staticIm.x,starts.staticIm.y,starts.staticIm.z,szs.x,szs.y,szs.z);
// 
// 						getROI<<<blocks,threads>>>(deviceOverlapImageSmooth,imSizes.overlapIm.x,imSizes.overlapIm.y,imSizes.overlapIm.z,
// 							deviceOverlapROIimage,starts.overlapIm.x,starts.overlapIm.y,starts.overlapIm.z,szs.x,szs.y,szs.z);
// 					}
// 					else
// 					{
						getROI<<<blocks,threads>>>(deviceStaticImage,imSizes.staticIm.x,imSizes.staticIm.y,imSizes.staticIm.z,
							deviceStaticROIimage,starts.staticIm.x,starts.staticIm.y,starts.staticIm.z,szs.x,szs.y,szs.z);

						getROI<<<blocks,threads>>>(deviceOverlapImage,imSizes.overlapIm.x,imSizes.overlapIm.y,imSizes.overlapIm.z,
							deviceOverlapROIimage,starts.overlapIm.x,starts.overlapIm.y,starts.overlapIm.z,szs.x,szs.y,szs.z);
					//}
					////////////////////////////////////////////////////////////

					double staticSig, overlapSig;
					float correlation = calcCorr(szs.x, szs.y, szs.z, prop, deviceStaticROIimage, deviceStaticSum, 
						deviceOverlapROIimage, deviceOverlapSum, staticSum, overlapSum, deviceMulImage, staticSig, overlapSig);

					report[reportInd.x + reportInd.y*deltaSizes.x + reportInd.z*deltaSizes.y*deltaSizes.x].delta = Vec<int>(deltaX,deltaY,deltaZ);
					report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].correlation = correlation;
					report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].staticSig = staticSig;
					report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].overlapSig = overlapSig;
					report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].nVoxels = szs.x*szs.y*szs.z;

					if (correlation>maxCorrelation)
					{
						maxCorrelation = correlation;
						bestDelta.x = deltaX;
						bestDelta.y = deltaY;
						bestDelta.z = deltaZ;
						bestN = szs.x*szs.y*szs.z;
					}
					++curIter;
					//  				time(&zEnd);
					//  				zSec = difftime(zEnd,zStart);
				}
				//  			time(&yEnd);
				//  			ySec = difftime(yEnd,yStart);
			}

			if (0==deltaX%5)
			{
				time(&xEnd);
				xSec = difftime(xEnd,xStart);

				printf("\t(%d)  BestCorr:%6.4f(%4d,%4d,%3d)",
					deviceNum, maxCorrelation, bestDelta.x, bestDelta.y, bestDelta.z);

				printf("  Done:%5.2f%% (%4d:%4d,%4d,%2d)",
					(float)curIter/iterations*100.0, deltaX, overlap.deltaXmax, overlap.deltaYmax,overlap.deltaZmax);

				printf("  X:%6.2f avgY:%5.3f avgZ:%6.4f Est(min):%6.2f\n",
					xSec, xSec/(overlap.deltaYmax-overlap.deltaYmin),
					xSec/((overlap.deltaYmax-overlap.deltaYmin)*(overlap.deltaZmax-overlap.deltaZmin)),
					(iterations-curIter)*(xSec/((overlap.deltaYmax-overlap.deltaYmin)*(overlap.deltaZmax-overlap.deltaZmin)))/60.0);
			}
		}
		time(&smoothEnd);
		smoothSec = difftime(smoothEnd,smoothStart);

		Vec<double> offsets;
		offsets.x = ((deltaMaxs.x-deltaMins.x)/3.0)/2.0;
		offsets.y = ((deltaMaxs.y-deltaMins.y)/3.0)/2.0;
		offsets.z = ((deltaMaxs.z-deltaMins.z)/3.0)/2.0;

		deltaMins.x = max(overlap.deltaXmin, (int)round(bestDelta.x - offsets.x));
		deltaMaxs.x = min(overlap.deltaXmax, (int)round(bestDelta.x + offsets.x));
		deltaMins.y = max(overlap.deltaYmin, (int)round(bestDelta.y - offsets.y));
		deltaMaxs.y = min(overlap.deltaYmax, (int)round(bestDelta.y + offsets.y));
		deltaMins.z = max(overlap.deltaZmin, (int)round(bestDelta.z - offsets.z));
		deltaMaxs.z = min(overlap.deltaZmax, (int)round(bestDelta.z + offsets.z));

		printf("   (%d) Delta (%d,%d,%d) max:%f totalTime:%f avgTime:%f\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,maxCorrelation,smoothSec,smoothSec/iterations);
		FILE* reportFile;
		fopen_s(&reportFile,fileName,"w");
		for (int i=0; i<reportInd.product(); ++i)
		{
			fprintf(reportFile,"(%d,%d,%d):%lf,%lf,%lf,%d\n",report[i].delta.x,report[i].delta.y,report[i].delta.z,
				report[i].correlation,report[i].staticSig,report[i].overlapSig,report[i].nVoxels);
		}
		fclose(reportFile);
	}
//////////////////////////////////////////////////////////////////////////
// 	comparedImages<int> starts;
// 	Vec<int> szs;
// 
// 	starts.staticIm.x = max(0,overlap.deltaXss+bestDelta.x-imStarts.staticIm.x);
// 	starts.overlapIm.x = max(0,-(overlap.deltaXss+bestDelta.x)-imStarts.overlapIm.x);
// 	szs.x = min(imSizes.staticIm.x,overlap.deltaXse+bestDelta.x+1) - starts.staticIm.x;
// 
// 	starts.staticIm.y = max(0,overlap.deltaYss+bestDelta.y-imStarts.staticIm.y);
// 	starts.overlapIm.y = max(0,-(overlap.deltaYss+bestDelta.y)-imStarts.overlapIm.y);
// 	szs.y = min(imSizes.staticIm.y,overlap.deltaYse+bestDelta.y+1) - starts.staticIm.y;
// 
// 	starts.staticIm.z = max(0,overlap.deltaZss+bestDelta.z-imStarts.staticIm.z);
// 	starts.overlapIm.z = max(0,-(overlap.deltaZss+bestDelta.z)-imStarts.overlapIm.z);
// 	szs.z = min(imSizes.staticIm.z,overlap.deltaZse+bestDelta.z+1) - starts.staticIm.z;

// 	getROI<<<blocks,threads>>>(deviceStaticImage,imSizes.staticIm.x,imSizes.staticIm.y,imSizes.staticIm.z,
// 		deviceStaticROIimage,starts.staticIm.x,starts.staticIm.y,starts.staticIm.z,szs.x,szs.y,szs.z);
// 
// 	getROI<<<blocks,threads>>>(deviceOverlapImage,imSizes.overlapIm.x,imSizes.overlapIm.y,imSizes.overlapIm.z,
// 		deviceOverlapROIimage,starts.overlapIm.x,starts.overlapIm.y,starts.overlapIm.z,szs.x,szs.y,szs.z);

// 	float* staticROI = new float[szs.x*szs.y*szs.z];
// 	float* overlapROI = new float[szs.x*szs.y*szs.z];
// 	HANDLE_ERROR(cudaMemcpy(staticROI,deviceStaticROIimage,sizeof(float)*szs.x*szs.y*szs.z,cudaMemcpyDeviceToHost));
// 	HANDLE_ERROR(cudaMemcpy(overlapROI,deviceOverlapROIimage,sizeof(float)*szs.x*szs.y*szs.z,cudaMemcpyDeviceToHost));
// 
// 
// 	writeImage(staticROI,szs.x,szs.y,szs.z,"staticROIsmoothed.tif");
// 	writeImage(overlapROI,szs.x,szs.y,szs.z,"overlapROIsmoothed.tif");
// 	delete staticROI;
// 	delete overlapROI;

//////////////////////////////////////////////////////////////////////////
	//Clean up
	HANDLE_ERROR(cudaFree(deviceStaticROIimage));
	HANDLE_ERROR(cudaFree(deviceOverlapROIimage));
	HANDLE_ERROR(cudaFree(deviceStaticSum));
	HANDLE_ERROR(cudaFree(deviceOverlapSum));
	HANDLE_ERROR(cudaFree(deviceMulImage));
	HANDLE_ERROR(cudaFree(deviceStaticImage));
	HANDLE_ERROR(cudaFree(deviceOverlapImage));
	//HANDLE_ERROR(cudaFree(deviceStaticImageSmooth));
	//HANDLE_ERROR(cudaFree(deviceOverlapImageSmooth));
	delete staticImageFloat;
	delete overlapImageFloat;
	delete staticSum;
	delete overlapSum;
//////////////////////////////////////////////////////////////////////////
	time(&mainEnd);
	mainSec = difftime(mainEnd,mainStart);

	printf("(%d) Delta (%d,%d,%d) max:%f totalTime:%f\n\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,maxCorrelation,mainSec);
	maxCorrOut = maxCorrelation;
}
*/