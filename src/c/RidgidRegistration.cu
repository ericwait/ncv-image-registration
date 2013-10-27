#include "RidgidRegistration.h"
#include "CudaProcessBuffer.cuh"
#include "CudaStorageBuffer.cuh"

struct  corrReport
{
	Vec<int> delta;
	double correlation;
	double staticSig;
	double overlapSig;
	unsigned int nVoxels;
};

void calcMaxROIs(const Overlap& overlap, Vec<int> imageExtents, comparedImages<unsigned int>& imStarts,
	comparedImages<unsigned int>& imSizes, Vec<unsigned int>& maxOverlapSize)
{
	comparedImages<int> localStarts;
	comparedImages<int> localSizes;
	localStarts.staticIm.x = std::numeric_limits<int>::max();
	localStarts.staticIm.y = std::numeric_limits<int>::max();
	localStarts.staticIm.z = std::numeric_limits<int>::max();
	localStarts.overlapIm.x = std::numeric_limits<int>::max();
	localStarts.overlapIm.y = std::numeric_limits<int>::max();
	localStarts.overlapIm.z = std::numeric_limits<int>::max();

	localSizes.staticIm.x = 0;
	localSizes.staticIm.y = 0;
	localSizes.staticIm.z = 0;
	localSizes.overlapIm.x = 0;
	localSizes.overlapIm.y = 0;
	localSizes.overlapIm.z = 0;

	for (int deltaX=overlap.deltaXmin; deltaX<overlap.deltaXmax; ++deltaX)
	{
		for (int deltaY=overlap.deltaYmin; deltaY<overlap.deltaYmax; ++deltaY)
		{
			for (int deltaZ=overlap.deltaZmin; deltaZ<overlap.deltaZmax; ++deltaZ)
			{
				comparedImages<int> mins;
				Vec<int> szs;

				mins.staticIm.x = std::max<int>(0,overlap.deltaXss+deltaX);
				mins.overlapIm.x = std::max<int>(0,-(overlap.deltaXss+deltaX));
				szs.x = std::min<int>(imageExtents.x,overlap.deltaXse+deltaX+1) - mins.staticIm.x;

				mins.staticIm.y = std::max<int>(0,overlap.deltaYss+deltaY);
				mins.overlapIm.y = std::max<int>(0,-(overlap.deltaYss+deltaY));
				szs.y = std::min<int>(imageExtents.y,overlap.deltaYse+deltaY+1) - mins.staticIm.y;

				mins.staticIm.z = std::max<int>(0,overlap.deltaZss+deltaZ);
				mins.overlapIm.z = std::max<int>(0,-(overlap.deltaZss+deltaZ));
				szs.z = std::min<int>(imageExtents.z,overlap.deltaZse+deltaZ+1) - mins.staticIm.z;

				localStarts.staticIm.x = std::min<int>(localStarts.staticIm.x,mins.staticIm.x);
				localStarts.staticIm.y = std::min<int>(localStarts.staticIm.y,mins.staticIm.y);
				localStarts.staticIm.z = std::min<int>(localStarts.staticIm.z,mins.staticIm.z);

				localSizes.staticIm.x = std::max<int>(localSizes.staticIm.x,mins.staticIm.x+szs.x);
				localSizes.staticIm.y = std::max<int>(localSizes.staticIm.y,mins.staticIm.y+szs.y);
				localSizes.staticIm.z = std::max<int>(localSizes.staticIm.z,mins.staticIm.z+szs.z);

				localStarts.overlapIm.x = std::min<int>(localStarts.overlapIm.x,mins.overlapIm.x);
				localStarts.overlapIm.y = std::min<int>(localStarts.overlapIm.y,mins.overlapIm.y);
				localStarts.overlapIm.z = std::min<int>(localStarts.overlapIm.z,mins.overlapIm.z);

				localSizes.overlapIm.x = std::max<int>(localSizes.overlapIm.x,mins.overlapIm.x+szs.x);
				localSizes.overlapIm.y = std::max<int>(localSizes.overlapIm.y,mins.overlapIm.y+szs.y);
				localSizes.overlapIm.z = std::max<int>(localSizes.overlapIm.z,mins.overlapIm.z+szs.z);

				maxOverlapSize.x = std::max<int>(maxOverlapSize.x,szs.x);
				maxOverlapSize.y = std::max<int>(maxOverlapSize.y,szs.y);
				maxOverlapSize.z = std::max<int>(maxOverlapSize.z,szs.z);
			}
		}
	}

	localSizes.staticIm.x -= localStarts.staticIm.x;
	localSizes.staticIm.y -= localStarts.staticIm.y;
	localSizes.staticIm.z -= localStarts.staticIm.z;
	localSizes.overlapIm.x -= localStarts.overlapIm.x;
	localSizes.overlapIm.y -= localStarts.overlapIm.y;
	localSizes.overlapIm.z -= localStarts.overlapIm.z;

	imStarts.staticIm.x = localStarts.staticIm.x;
	imStarts.staticIm.y = localStarts.staticIm.y;
	imStarts.staticIm.z = localStarts.staticIm.z;
	imStarts.overlapIm.x = localStarts.overlapIm.x;
	imStarts.overlapIm.y = localStarts.overlapIm.y;
	imStarts.overlapIm.z = localStarts.overlapIm.z;
	imSizes.staticIm.x = localSizes.staticIm.x;
	imSizes.staticIm.y = localSizes.staticIm.y;
	imSizes.staticIm.z = localSizes.staticIm.z;
	imSizes.overlapIm.x = localSizes.overlapIm.x;
	imSizes.overlapIm.y = localSizes.overlapIm.y;
	imSizes.overlapIm.z = localSizes.overlapIm.z;
}

#pragma  optimize("",off)
void findRegistration(CudaStorageBuffer<float> &staticMaxRoiCuda, CudaStorageBuffer<float> &overlapMaxRoiCuda,
					  CudaProcessBuffer<float> &staticCudaIm, CudaProcessBuffer<float> &overlapCudaIm, int deviceNum,
					  const Vec<int> deltaMins, const Vec<int> deltaMaxs, const Overlap overlap,
					  const comparedImages<unsigned int> imStarts, const comparedImages<unsigned int> imSizes,
					  /*returns*/ double &maxCorrOut, Vec<int> &bestDelta, unsigned int &bestN)
{
	volatile double xSecTotal=0;
	unsigned int curIter=0;
	comparedImages<unsigned int> starts, szs;
	int iterations = (deltaMaxs-deltaMins).product();

	Vec<int> curDeltas;
	for (curDeltas.x=deltaMins.x; curDeltas.x<deltaMaxs.x; ++curDeltas.x)
	{
		time_t xStart, xEnd;
		time(&xStart);
		for (curDeltas.y=deltaMins.y; curDeltas.y<deltaMaxs.y; ++curDeltas.y)
		{
			for (curDeltas.z=deltaMins.z; curDeltas.z<deltaMaxs.z; ++curDeltas.z)
			{
				starts.staticIm.x = (unsigned int)std::max<int>(0,overlap.deltaXss+curDeltas.x-imStarts.staticIm.x);
				starts.overlapIm.x = (unsigned int)std::max<int>(0,-(overlap.deltaXss+curDeltas.x)-imStarts.overlapIm.x);
				szs.staticIm.x = szs.overlapIm.x = (unsigned int)std::min<int>(imSizes.staticIm.x,overlap.deltaXse+curDeltas.x+1) - starts.staticIm.x;

				starts.staticIm.y = (unsigned int)std::max<int>(0,overlap.deltaYss+curDeltas.y-imStarts.staticIm.y);
				starts.overlapIm.y = (unsigned int)std::max<int>(0,-(overlap.deltaYss+curDeltas.y)-imStarts.overlapIm.y);
				szs.staticIm.y = szs.overlapIm.y = (unsigned int)std::min<int>(imSizes.staticIm.y,overlap.deltaYse+curDeltas.y+1) - starts.staticIm.y;

				starts.staticIm.z = (unsigned int)std::max<int>(0,overlap.deltaZss+curDeltas.z-imStarts.staticIm.z);
				starts.overlapIm.z = (unsigned int)std::max<int>(0,-(overlap.deltaZss+curDeltas.z)-imStarts.overlapIm.z);
				szs.staticIm.z = szs.overlapIm.z = (unsigned int)std::min<int>(imSizes.staticIm.z,overlap.deltaZse+curDeltas.z+1) - starts.staticIm.z;

				staticCudaIm.copyROI(&staticMaxRoiCuda,starts.staticIm,szs.staticIm);
				overlapCudaIm.copyROI(&overlapMaxRoiCuda,starts.overlapIm,szs.overlapIm);

				double curCorr = staticCudaIm.normalizedCovariance(&overlapCudaIm);
				if (curCorr>1)
				{
					printf("Warning: Got a correlation greater than 1!\n");
					curCorr = 0;
				}else if (curCorr<-1)
				{
					printf("Warning: Got a correlation less than -1!\n");
				}

				if (curCorr>maxCorrOut)
				{
					maxCorrOut = curCorr;
					bestDelta.x = curDeltas.x;
					bestDelta.y = curDeltas.y;
					bestDelta.z = curDeltas.z;
					bestN = szs.staticIm.x*szs.staticIm.y;
				}
				++curIter; 		

				float progress = (float)curIter/iterations*100.0f;
				int printThresh = 10000;
#ifdef _DEBUG
				printThresh = 5000;
#endif // _DEBUG
				if ((int)ceil(progress*1000)%printThresh==0)
				{
					for (int i=0; i<deviceNum; ++i)
						printf("  ");

					printf("  (%d)  BestCorr:%6.4f(%4d,%4d,%3d)", deviceNum, maxCorrOut, bestDelta.x, bestDelta.y, bestDelta.z);
					printf(" Done:%5.2f%%  ", progress);
					double est = (iterations-curIter)*(xSecTotal/curIter);
					int estMin = (int)floor(est/60.0);
					int estSec = (int)floor(est)%60;
					printf("Est(min):%d:%02d\n", estMin, estSec);
				}
			}
		}
		time(&xEnd);
		volatile double xSec = difftime(xEnd,xStart);
		xSecTotal += xSec;		
	}
}

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap,
	Vec<int>& bestDelta, double& maxCorrOut, unsigned int& bestN, int deviceNum, const char* fileName)
{
	bestDelta = Vec<int>(0,0,0);
	maxCorrOut = std::numeric_limits<double>::min();
	bestN = 0;

	if (staticImage==NULL || overlapImage==NULL)
		return;

	comparedImages<unsigned int> imStarts, imSizes;
	Vec<unsigned int> maxOverlapSize;
	Vec<int> staticImageExtents(staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth());

	calcMaxROIs(overlap,staticImageExtents,imStarts,imSizes,maxOverlapSize);
	if (imSizes.staticIm>=staticImage->getDims())
		fprintf(stderr,"Using Total Static Image Size!\n");
	if (imSizes.overlapIm>=overlapImage->getDims())
		fprintf(stderr,"Using Total Overlap Image Size!\n");

	time_t mainStart, mainEnd, mipStart, mipEnd, zStart, zEnd;
	Vec<int> deltaMins(overlap.deltaXmin,overlap.deltaYmin,overlap.deltaZmin);
	Vec<int> deltaMaxs(overlap.deltaXmax,overlap.deltaYmax,overlap.deltaZmax);

	time(&mainStart);

	const float* staticMaxRoi = staticImage->getFloatConstROIData(imStarts.staticIm,imSizes.staticIm);
	const float* overlapMaxRoi = overlapImage->getFloatConstROIData(imStarts.overlapIm,imSizes.overlapIm);

	CudaStorageBuffer<float> staticMaxRoiCuda(staticMaxRoi,imSizes.staticIm,deviceNum);
	CudaStorageBuffer<float> overlapMaxRoiCuda(overlapMaxRoi,imSizes.overlapIm,deviceNum);

	const size_t MAX_MEM = staticMaxRoiCuda.getGlobalMemoryAvailable();
	size_t memUsed = sizeof(float)*
		((NUM_BUFFERS+1)*(imSizes.staticIm.product()+imSizes.overlapIm.product()+imSizes.staticIm.x*imSizes.staticIm.y+
		 imSizes.overlapIm.x*imSizes.overlapIm.y));
	
	printf("(%d) Deltas(%d to %d, %d to %d, %d to %d) Max Overlap(%d, %d, %d) Memory Needed(%04.2f%%, %6.2fMB, %dMB)\n",deviceNum,
		deltaMins.x,deltaMaxs.x,deltaMins.y,deltaMaxs.y,deltaMins.z,deltaMaxs.z,
		maxOverlapSize.x,maxOverlapSize.y,maxOverlapSize.z,
		(float)memUsed/MAX_MEM*100.0f,(float)memUsed/1024.0f/1024.0f,(int)(MAX_MEM/1024.0f/1024.0f));

	if (memUsed>=MAX_MEM*0.8)
	{
		printf("\n(%d) Overlap TOO BIG!\n\n",deviceNum);
		return;
	}

	CudaProcessBuffer<float> staticCudaIm(maxOverlapSize,false,deviceNum);
	CudaProcessBuffer<float> overlapCudaIm(maxOverlapSize,false,deviceNum);
	staticCudaIm.copyROI(&staticMaxRoiCuda,Vec<unsigned int>(0,0,0),imSizes.staticIm);
	overlapCudaIm.copyROI(&overlapMaxRoiCuda,Vec<unsigned int>(0,0,0),imSizes.overlapIm);

	staticCudaIm.maximumIntensityProjection();
	overlapCudaIm.maximumIntensityProjection();

	CudaStorageBuffer<float> staticMIP(&staticCudaIm);
	CudaStorageBuffer<float> overlapMIP(&overlapCudaIm);

	comparedImages<unsigned int> imMIPstarts=imStarts, imMIPsizes=imSizes;

	imMIPstarts.overlapIm.z = 0;
	imMIPstarts.staticIm.z = 0;
	imMIPsizes.overlapIm.z = 1;
	imMIPsizes.staticIm.z = 1;
	deltaMins.z = 0;
	deltaMaxs.z = 1;

	time(&mipStart);

	findRegistration(staticMIP,overlapMIP,staticCudaIm,overlapCudaIm,deviceNum,deltaMins,deltaMaxs,overlap,imMIPstarts,imMIPsizes,
		maxCorrOut,bestDelta,bestN);

	time(&mipEnd);
	volatile double mipTime = difftime(mipEnd,mipStart);

	int mipMin = (int)floor(mipTime/60.0);
	int mipSec = (int)floor(mipTime)%60;

	printf("\n");

	for (int i=0; i<deviceNum; ++i)
		printf("  ");

	printf("  (%d) *Delta (%d,%d,%d) max:%f mipsTotalTime(min):%d:%02d avgTime:%5.5f\n\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,
		maxCorrOut,mipMin,mipSec,mipTime/(deltaMaxs-deltaMins).product());

	deltaMins.x = bestDelta.x-LOCAL_REGION;
	deltaMaxs.x = bestDelta.x+LOCAL_REGION;
	deltaMins.y = bestDelta.y-LOCAL_REGION;
	deltaMaxs.y = bestDelta.y+LOCAL_REGION;
	deltaMins.z = overlap.deltaZmin;
	deltaMaxs.z = overlap.deltaZmax;

	maxCorrOut = -std::numeric_limits<double>::infinity();

	time(&zStart);
	findRegistration(staticMaxRoiCuda,overlapMaxRoiCuda,staticCudaIm,overlapCudaIm,deviceNum,deltaMins,deltaMaxs,overlap,imStarts,imSizes,
		maxCorrOut,bestDelta,bestN);

	time(&zEnd);
	volatile double zTime = difftime(zEnd,zStart);

	int zMins = (int)floor(zTime/60.0);
	int zSecs = (int)zTime%60;

	printf("\n");

	for (int i=0; i<deviceNum; ++i)
		printf("  ");

	printf("  (%d) **Delta (%d,%d,%d) max:%f zTotalTime(min):%d:%02d\n\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,
		maxCorrOut,zMins,zSecs);

	time(&mainEnd);
	volatile double mainTime = difftime(mainEnd,mainStart);

	int totMin = (int)floor(mainTime/60.0);
	int totSec = (int)floor(mainTime)%60;

	for (int i=0; i<deviceNum; ++i)
		printf("  ");

	printf("  (%d) ***Delta (%d,%d,%d) max:%5.5f totalTime(min):%d:%02d\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,maxCorrOut,totMin,totSec);
}
