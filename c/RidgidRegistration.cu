#include "RidgidRegistration.h"
#include "CudaImageBuffer.cuh"

struct  corrReport
{
	Vec<int> delta;
	float correlation;
	float staticSig;
	float overlapSig;
	unsigned int nVoxels;
};

float calcCorrelation(CudaImageBuffer<float>& staticIm, CudaImageBuffer<float>& overlapIm, CudaImageBuffer<float>& multiplyIm,
	float& staticSigma, float& overlapSigma)
{
	float staticSum = 0;
	float overlapSum = 0;

	staticIm.sumArray(staticSum);
	overlapIm.sumArray(overlapSum);

	float staticMean = staticSum/staticIm.getDimension().product();
	float overlapMean = overlapSum/overlapIm.getDimension().product();

	staticIm.addConstant(-staticMean);
	overlapIm.addConstant(-overlapMean);

	multiplyIm.copyImage(overlapIm);
	multiplyIm.multiplyImageWith(&staticIm);

	staticIm.pow(2);
	overlapIm.pow(2);

	staticIm.sumArray(staticSum);
	overlapIm.sumArray(overlapSum);

	staticSigma = sqrt(staticSum/staticIm.getDimension().product());
	overlapSigma = sqrt(overlapSum/overlapIm.getDimension().product());

	float multSum = 0.0;
	multiplyIm.sumArray(multSum);

	return multSum/(staticSigma*overlapSigma) / staticIm.getDimension().product();
}

void calcMaxROIs(const Overlap& overlap, Vec<int> imageExtents, comparedImages<unsigned int>& imStarts,
	comparedImages<unsigned int>& imSizes, Vec<int>& maxOverlapSize)
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

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap,
	Vec<int>& bestDelta, float& maxCorrOut, unsigned int& bestN, int deviceNum, const char* fileName)
{
	bestDelta = Vec<int>(0,0,0);
	maxCorrOut = std::numeric_limits<float>::min();
	bestN = 0;

	if (staticImage==NULL || overlapImage==NULL)
		return;

	comparedImages<unsigned int> imStarts, imSizes;
	Vec<int> maxOverlapSize;
	Vec<int> staticImageExtents(staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth());

	calcMaxROIs(overlap,staticImageExtents,imStarts,imSizes,maxOverlapSize);
	if (imSizes.staticIm>=staticImage->getDims())
		fprintf(stderr,"Using Total Static Image Size!\n");
	if (imSizes.overlapIm>=overlapImage->getDims())
		fprintf(stderr,"Using Total Overlap Image Size!\n");

	time_t mainStart, mainEnd, mipStart, mipEnd, xStart, xEnd, zStart, zEnd;
	double mainSec=0, mipSec=0, xSec=0, xSecTotal=0.0, zSec=0, zSecTotal=0.0;
	Vec<int> deltaMins(overlap.deltaXmin,overlap.deltaYmin,overlap.deltaZmin);
	Vec<int> deltaMaxs(overlap.deltaXmax,overlap.deltaYmax,overlap.deltaZmax);

	time(&mainStart);

	unsigned int iterations = std::max<int>(1,(deltaMaxs.x-deltaMins.x)*(deltaMaxs.y-deltaMins.y));
	unsigned int curIter = 0;
	Vec<int> deltaSizes((deltaMaxs.x-deltaMins.x),(deltaMaxs.y-deltaMins.y),(deltaMaxs.z-deltaMins.z));

	//corrReport* report = new corrReport[deltaSizes.product()];
	//Vec<int> reportInd(0,0,0);

	comparedImages<unsigned int> starts;
	comparedImages<unsigned int> szs;

	printf("(%d) Deltas(%d to %d, %d to %d, %d to %d) Sizes(%d, %d, %d)\n",deviceNum,
		deltaMins.x,deltaMaxs.x,deltaMins.y,deltaMaxs.y,deltaMins.z,deltaMaxs.z,
		maxOverlapSize.x,maxOverlapSize.y,maxOverlapSize.z);

	//Start of MIPs registration
	{
		const float* staticMaxRoi = staticImage->getFloatConstROIData(imStarts.staticIm,imSizes.staticIm);
		const float* overlapMaxRoi = overlapImage->getFloatConstROIData(imStarts.overlapIm,imSizes.overlapIm);

		CudaImageBuffer<float> staticMaxRoiCuda(imSizes.staticIm,deviceNum);
		CudaImageBuffer<float> overlapMaxRoiCuda(imSizes.overlapIm,deviceNum);

		staticMaxRoiCuda.loadImage(staticMaxRoi);
		overlapMaxRoiCuda.loadImage(overlapMaxRoi);

		staticMaxRoiCuda.maximumIntensityProjection();
		overlapMaxRoiCuda.maximumIntensityProjection();

		CudaImageBuffer<float> staticCudaIm(staticMaxRoiCuda.getDimension(),deviceNum);
		CudaImageBuffer<float> overlapCudaIm(overlapMaxRoiCuda.getDimension(),deviceNum);
		CudaImageBuffer<float> multiplyCudaIm(MAX(overlapMaxRoiCuda.getDimension(),staticMaxRoiCuda.getDimension()),deviceNum);

		time(&mipStart);
		for (int deltaX=deltaMins.x; deltaX<deltaMaxs.x; ++deltaX)//, ++reportInd.x)
		{
			//reportInd.y = 0;
			time(&xStart);
			for (int deltaY=deltaMins.y; deltaY<deltaMaxs.y; ++deltaY)//, ++reportInd.y)
			{
				starts.staticIm.x = (unsigned int)std::max<int>(0,overlap.deltaXss+deltaX-imStarts.staticIm.x);
				starts.overlapIm.x = (unsigned int)std::max<int>(0,-(overlap.deltaXss+deltaX)-imStarts.overlapIm.x);
				szs.staticIm.x = szs.overlapIm.x = (unsigned int)std::min<int>(imSizes.staticIm.x,overlap.deltaXse+deltaX+1) - starts.staticIm.x;

				starts.staticIm.y = (unsigned int)std::max<int>(0,overlap.deltaYss+deltaY-imStarts.staticIm.y);
				starts.overlapIm.y = (unsigned int)std::max<int>(0,-(overlap.deltaYss+deltaY)-imStarts.overlapIm.y);
				szs.staticIm.y = szs.overlapIm.y = (unsigned int)std::min<int>(imSizes.staticIm.y,overlap.deltaYse+deltaY+1) - starts.staticIm.y;
				szs.staticIm.z = 1;
				szs.overlapIm.z = 1;

				staticCudaIm.copyROI(staticMaxRoiCuda,starts.staticIm,szs.staticIm);
				overlapCudaIm.copyROI(overlapMaxRoiCuda,starts.overlapIm,szs.overlapIm);

				float staticSigma, overlapSigma;
				float curCorr = calcCorrelation(staticCudaIm,overlapCudaIm,multiplyCudaIm,staticSigma,overlapSigma);

// 				report[deltaSizes.linearAddressAt(reportInd)].delta = Vec<int>(deltaX,deltaY,0);
// 				report[deltaSizes.linearAddressAt(reportInd)].correlation = curCorr;
// 				report[deltaSizes.linearAddressAt(reportInd)].staticSig = staticSigma;
// 				report[deltaSizes.linearAddressAt(reportInd)].overlapSig = overlapSigma;
// 				report[deltaSizes.linearAddressAt(reportInd)].nVoxels = szs.staticIm.x*szs.staticIm.y;

				if (curCorr>maxCorrOut)
				{
					maxCorrOut = curCorr;
					bestDelta.x = deltaX;
					bestDelta.y = deltaY;
					bestDelta.z = 0;
					bestN = szs.staticIm.x*szs.staticIm.y;
				}
				++curIter; 			
			}
			time(&xEnd);
			xSec = difftime(xEnd,xStart);
			xSecTotal += xSec;

#ifndef _DEBUG
			if (0==deltaX%20)
#endif // _DEBUG
			{
				printf("\t");
				for (int i=0; i<deviceNum; ++i)
					printf("  ");

				printf("(%d)  BestCorr:%6.4f(%4d,%4d,%3d)", deviceNum, maxCorrOut, bestDelta.x, bestDelta.y, bestDelta.z);
				printf(" Done:%5.2f%% deltaX= %+03d", (float)curIter/iterations*100.0, deltaX);
				double est = (iterations-curIter)*(xSecTotal/(deltaSizes.y*(abs(deltaMins.x-deltaX)+1)));
				printf(" X(sec):%4.1f avgY(sec):%5.3f Est(min):%d:%02d\n", xSec, xSec/deltaSizes.y, floor(est/60.0),(int)est%60);
			}
		}

		delete[] staticMaxRoi;
		delete[] overlapMaxRoi;
	}//End of MIPs registration

	time(&mipEnd);
	mipSec = difftime(mipEnd,mipStart);

	Vec<double> offsets;
	offsets.x = ((deltaMaxs.x-deltaMins.x)/3.0)/2.0;
	offsets.y = ((deltaMaxs.y-deltaMins.y)/3.0)/2.0;
	offsets.z = ((deltaMaxs.z-deltaMins.z)/3.0)/2.0;

	deltaMins.x = std::max<int>(overlap.deltaXmin, (int)round((double)bestDelta.x - offsets.x));
	deltaMaxs.x = std::min<int>(overlap.deltaXmax, (int)round((double)bestDelta.x + offsets.x));
	deltaMins.y = std::max<int>(overlap.deltaYmin, (int)round((double)bestDelta.y - offsets.y));
	deltaMaxs.y = std::min<int>(overlap.deltaYmax, (int)round((double)bestDelta.y + offsets.y));
	deltaMins.z = std::max<int>(overlap.deltaZmin, (int)round((double)bestDelta.z - offsets.z));
	deltaMaxs.z = std::min<int>(overlap.deltaZmax, (int)round((double)bestDelta.z + offsets.z));

	printf("   (%d) Delta (%d,%d,%d) max:%f mipsTotalTime(min):%d:%02d avgTime:%5.5f\n\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,maxCorrOut,floor(mipSec/60.0),(int)mipSec%60,mipSec/iterations);

	iterations = (deltaMaxs.z-deltaMins.z)*SQR(LOCAL_REGION*2+1);
	curIter = 0;
	maxCorrOut = -std::numeric_limits<double>::infinity();

	//Start of Z stack registration
	{
		const float* staticMaxRoi = staticImage->getFloatConstROIData(imStarts.staticIm,imSizes.staticIm);
		const float* overlapMaxRoi = overlapImage->getFloatConstROIData(imStarts.overlapIm,imSizes.overlapIm);

		CudaImageBuffer<float> staticMaxRoiCuda(imSizes.staticIm,deviceNum);
		CudaImageBuffer<float> overlapMaxRoiCuda(imSizes.overlapIm,deviceNum);

		staticMaxRoiCuda.loadImage(staticMaxRoi);
		overlapMaxRoiCuda.loadImage(overlapMaxRoi);

		CudaImageBuffer<float> staticCudaIm(staticMaxRoiCuda.getDimension(),deviceNum);
		CudaImageBuffer<float> overlapCudaIm(overlapMaxRoiCuda.getDimension(),deviceNum);
		CudaImageBuffer<float> multiplyCudaIm(MAX(overlapMaxRoiCuda.getDimension(),staticMaxRoiCuda.getDimension()),deviceNum);

		//reportInd.z = 0;
		for (int deltaZ=deltaMins.z; deltaZ<deltaMaxs.z; ++deltaZ)//, ++reportInd.z)
		{
			time(&zStart);
			//reportInd.x = bestDelta.x - deltaMins.x;
			for (int deltaX=bestDelta.x-LOCAL_REGION; deltaX<bestDelta.x+LOCAL_REGION; ++deltaX)//, ++reportInd.x)
			{
				//reportInd.y = bestDelta.y - deltaMins.y;
				for (int deltaY=bestDelta.y-LOCAL_REGION; deltaY<bestDelta.y+LOCAL_REGION; ++deltaY)//, ++reportInd.y)
				{
					starts.staticIm.x = (unsigned int)std::max<int>(0,overlap.deltaXss+deltaX-imStarts.staticIm.x);
					starts.overlapIm.x = (unsigned int)std::max<int>(0,-(overlap.deltaXss+deltaX)-imStarts.overlapIm.x);
					szs.staticIm.x = szs.overlapIm.x = (unsigned int)std::min<int>(imSizes.staticIm.x,overlap.deltaXse+deltaX+1) - starts.staticIm.x;

					starts.staticIm.y = (unsigned int)std::max<int>(0,overlap.deltaYss+deltaY-imStarts.staticIm.y);
					starts.overlapIm.y = (unsigned int)std::max<int>(0,-(overlap.deltaYss+deltaY)-imStarts.overlapIm.y);
					szs.staticIm.y = szs.overlapIm.y = (unsigned int)std::min<int>(imSizes.staticIm.y,overlap.deltaYse+deltaY+1) - starts.staticIm.y;

					starts.staticIm.z = (unsigned int)std::max<int>(0,overlap.deltaZss+deltaZ-imStarts.staticIm.z);
					starts.overlapIm.z = (unsigned int)std::max<int>(0,-(overlap.deltaZss+deltaZ)-imStarts.overlapIm.z);
					szs.staticIm.z = szs.overlapIm.z = (unsigned int)std::min<int>(imSizes.staticIm.z,overlap.deltaZse+deltaZ+1) - starts.staticIm.z;
					staticCudaIm.copyROI(staticMaxRoiCuda,starts.staticIm,szs.staticIm);
					overlapCudaIm.copyROI(overlapMaxRoiCuda,starts.overlapIm,szs.overlapIm);

					float staticSigma, overlapSigma;
					float curCorr = calcCorrelation(staticCudaIm,overlapCudaIm,multiplyCudaIm,staticSigma,overlapSigma);

// 					report[deltaSizes.linearAddressAt(reportInd)].delta = Vec<int>(deltaX,deltaY,deltaZ);
// 					report[deltaSizes.linearAddressAt(reportInd)].correlation = curCorr;
// 					report[deltaSizes.linearAddressAt(reportInd)].staticSig = staticSigma;
// 					report[deltaSizes.linearAddressAt(reportInd)].overlapSig = overlapSigma;
// 					report[deltaSizes.linearAddressAt(reportInd)].nVoxels = (unsigned int)szs.staticIm.product();

					if (curCorr>maxCorrOut)
					{
						maxCorrOut = curCorr;
						bestDelta.z = deltaZ;
						bestN = (unsigned int)szs.staticIm.product();
					}
					++curIter;
				}
			}
			time(&zEnd);
			zSec = difftime(zEnd,zStart);
			zSecTotal += zSec;

//#ifndef _DEBUG
			if (0==deltaZ%4)
//#endif
			{
				printf("\t");
				for (int i=0; i<deviceNum; ++i)
					printf("  ");

				printf("(%d)  BestCorr:%6.4f(%4d,%4d,%3d)", deviceNum, maxCorrOut, bestDelta.x, bestDelta.y, bestDelta.z);
				printf(" Done:%5.2f%% deltaZ= %+02d", (float)curIter/iterations*100.0, deltaZ);
				double iterationsLeft = iterations-curIter;
				double currentZiterations = (deltaZ-deltaMins.z)+1;
				double numOfinnerIterations = SQR(LOCAL_REGION*2.0+1.0);
				double est = iterationsLeft*(zSecTotal/(currentZiterations*numOfinnerIterations));
				printf(" Z(sec):%4.1f avgX(sec):%5.3f avgY(sec):%5.3f Est(min):%d:%02d\n", zSec, zSec/(2*LOCAL_REGION +1),zSec/SQR(2*LOCAL_REGION+1), floor(est/60.0),(int)est%60);
			}
		}
		delete[] staticMaxRoi;
		delete[] overlapMaxRoi;
	}//End of Z stack registration

	time(&mainEnd);
	mainSec = difftime(mainEnd,mainStart);

	printf("  (%d) Delta (%d,%d,%d) max:%5.5f totalTime(min):%d:%02d\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,maxCorrOut,floor(mainSec/60.0),(int)mainSec%60);
// 	FILE* reportFile;
// 	fopen_s(&reportFile,fileName,"w");
// 	for (int i=0; i<reportInd.product(); ++i)
// 	{
// 		fprintf(reportFile,"(%d,%d,%d):%lf,%lf,%lf,%d\n",report[i].delta.x,report[i].delta.y,report[i].delta.z,
// 			report[i].correlation,report[i].staticSig,report[i].overlapSig,report[i].nVoxels);
// 	}
// 	fclose(reportFile);

	//delete[] report;
}
