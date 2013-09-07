#include "RidgidRegistration.h"
#include "CudaImageBuffer.cuh"

struct  corrReport
{
	Vec<int> delta;
	double correlation;
	double staticSig;
	double overlapSig;
	unsigned int nVoxels;
};

double calcCorrelation(CudaImageBuffer<float>& staticIm, CudaImageBuffer<float>& overlapIm, double& staticSigma, double& overlapSigma)
{
	double staticSum = 0.0;
	double overlapSum = 0.0;

	staticIm.sumArray(staticSum);
	overlapIm.sumArray(overlapSum);

	double staticMean = staticSum/staticIm.getDimension().product();
	double overlapMean = overlapSum/overlapIm.getDimension().product();

	staticIm.addConstant(-staticMean);
	overlapIm.addConstant(-overlapMean);

	CudaImageBuffer<float> multIm = staticIm;
	multIm.multiplyImageWith(&overlapIm);

	staticIm.pow(2.0f);
	overlapIm.pow(2.0f);

	staticIm.sumArray(staticSum);
	overlapIm.sumArray(overlapSum);

	staticSigma = sqrt(staticSum/staticIm.getDimension().product());
	overlapSigma = sqrt(overlapSum/overlapIm.getDimension().product());

	double multSum = 0.0;
	multIm.sumArray(multSum);

	return multSum/(staticSigma*overlapSigma);
}

void calcMaxROIs(const Overlap& overlap, Vec<int> imageExtents, comparedImages<unsigned int>& imStarts,
	comparedImages<unsigned int>& imSizes, Vec<int>& maxOverlapSize)
{
	imStarts.staticIm.x = (int)(pow(2.0,(int)sizeof(int)));
	imStarts.staticIm.y = (int)(pow(2.0,(int)sizeof(int)));
	imStarts.staticIm.z = (int)(pow(2.0,(int)sizeof(int)));
	imStarts.overlapIm.x = (int)(pow(2.0,(int)sizeof(int)));
	imStarts.overlapIm.y = (int)(pow(2.0,(int)sizeof(int)));
	imStarts.overlapIm.z = (int)(pow(2.0,(int)sizeof(int)));

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
				comparedImages<unsigned int> mins;
				Vec<int> szs;

				mins.staticIm.x = (unsigned int)std::max<int>(0,overlap.deltaXss+deltaX);
				mins.overlapIm.x = (unsigned int)std::max<int>(0,-(overlap.deltaXss+deltaX));
				szs.x = std::min<int>(imageExtents.x,overlap.deltaXse+deltaX+1) - mins.staticIm.x;

				mins.staticIm.y = (unsigned int)std::max<int>(0,overlap.deltaYss+deltaY);
				mins.overlapIm.y = (unsigned int)std::max<int>(0,-(overlap.deltaYss+deltaY));
				szs.y = std::min<int>(imageExtents.x,overlap.deltaYse+deltaY+1) - mins.staticIm.y;

				mins.staticIm.z = (unsigned int)std::max<int>(0,overlap.deltaZss+deltaZ);
				mins.overlapIm.z = (unsigned int)std::max<int>(0,-(overlap.deltaZss+deltaZ));
				szs.z = std::min<int>(imageExtents.z,overlap.deltaZse+deltaZ+1) - mins.staticIm.z;

				imStarts.staticIm.x = (unsigned int)std::min<int>(imStarts.staticIm.x,mins.staticIm.x);
				imStarts.staticIm.y = (unsigned int)std::min<int>(imStarts.staticIm.y,mins.staticIm.y);
				imStarts.staticIm.z = (unsigned int)std::min<int>(imStarts.staticIm.z,mins.staticIm.z);

				imSizes.staticIm.x = (unsigned int)std::max<int>(imSizes.staticIm.x,mins.staticIm.x+szs.x);
				imSizes.staticIm.y = (unsigned int)std::max<int>(imSizes.staticIm.y,mins.staticIm.y+szs.y);
				imSizes.staticIm.z = (unsigned int)std::max<int>(imSizes.staticIm.z,mins.staticIm.z+szs.z);

				imStarts.overlapIm.x = (unsigned int)std::min<int>(imStarts.overlapIm.x,mins.overlapIm.x);
				imStarts.overlapIm.y = (unsigned int)std::min<int>(imStarts.overlapIm.y,mins.overlapIm.y);
				imStarts.overlapIm.z = (unsigned int)std::min<int>(imStarts.overlapIm.z,mins.overlapIm.z);

				imSizes.overlapIm.x = (unsigned int)std::max<int>(imSizes.overlapIm.x,mins.overlapIm.x+szs.x);
				imSizes.overlapIm.y = (unsigned int)std::max<int>(imSizes.overlapIm.y,mins.overlapIm.y+szs.y);
				imSizes.overlapIm.z = (unsigned int)std::max<int>(imSizes.overlapIm.z,mins.overlapIm.z+szs.z);

				maxOverlapSize.x = std::max<int>(maxOverlapSize.x,szs.x);
				maxOverlapSize.y = std::max<int>(maxOverlapSize.y,szs.y);
				maxOverlapSize.z = std::max<int>(maxOverlapSize.z,szs.z);
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

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap,
	Vec<int>& bestDelta, double& maxCorrOut, unsigned int& bestN, int deviceNum, const char* fileName)
{
	comparedImages<unsigned int> imStarts, imSizes;
	Vec<int> maxOverlapSize;
	Vec<int> staticImageExtents(staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth());

	calcMaxROIs(overlap,staticImageExtents,imStarts,imSizes,maxOverlapSize);

	time_t mainStart, mainEnd, mipStart, mipEnd, xStart, xEnd;//, yStart, yEnd, zStart, zEnd;
	double mainSec=0, mipSec=0, xSec=0;//, ySec=0, zSec=0;
	Vec<int> deltaMins(overlap.deltaXmin,overlap.deltaYmin,overlap.deltaZmin);
	Vec<int> deltaMaxs(overlap.deltaXmax,overlap.deltaYmax,overlap.deltaZmax);
	double maxCorrelation = -std::numeric_limits<double>::infinity();

	time(&mainStart);

	unsigned int iterations = std::max<int>(1,(deltaMaxs.x-deltaMins.x)*(deltaMaxs.y-deltaMins.y));
	unsigned int curIter = 0;
	Vec<int> deltaSizes((deltaMaxs.x-deltaMins.x),(deltaMaxs.y-deltaMins.y),(deltaMaxs.z-deltaMins.z));

	corrReport* report = new corrReport[deltaSizes.product()];
	Vec<int> reportInd(0,0,0);

	comparedImages<unsigned int> starts;
	comparedImages<unsigned int> szs;

	time(&mipStart);
	for (int deltaX=deltaMins.x; deltaX<deltaMaxs.x; ++deltaX, ++reportInd.x)
	{
		reportInd.y = 0;
		time(&xStart);
		for (int deltaY=deltaMins.y; deltaY<deltaMaxs.y; ++deltaY, ++reportInd.y)
		{
			starts.staticIm.x = (unsigned int)std::max<int>(0,overlap.deltaXss+deltaX-imStarts.staticIm.x);
			starts.overlapIm.x = (unsigned int)std::max<int>(0,-(overlap.deltaXss+deltaX)-imStarts.overlapIm.x);
			szs.staticIm.x = szs.overlapIm.x = (unsigned int)std::min<int>(imSizes.staticIm.x,overlap.deltaXse+deltaX+1) - starts.staticIm.x;

			starts.staticIm.y = (unsigned int)std::max<int>(0,overlap.deltaYss+deltaY-imStarts.staticIm.y);
			starts.overlapIm.y = (unsigned int)std::max<int>(0,-(overlap.deltaYss+deltaY)-imStarts.overlapIm.y);
			szs.staticIm.y = szs.overlapIm.y = (unsigned int)std::min<int>(imSizes.staticIm.y,overlap.deltaYse+deltaY+1) - starts.staticIm.y;
			szs.staticIm.z = staticImage->getDepth();
			szs.overlapIm.z = overlapImage->getDepth();

			const float* staticRoi = staticImage->getFloatConstROIData(starts.staticIm,szs.staticIm);
			const float* overlapRoi = overlapImage->getFloatConstROIData(starts.overlapIm,szs.overlapIm);

			CudaImageBuffer<float> staticCudaIm(szs.staticIm,deviceNum);
			CudaImageBuffer<float> overlapCudaIm(szs.overlapIm,deviceNum);

			staticCudaIm.loadImage(staticRoi);
			overlapCudaIm.loadImage(overlapRoi);

			float* staticPrintF = new float[staticCudaIm.getDimension().product()];
			float* overlapPrintf = new float[overlapCudaIm.getDimension().product()];
			staticCudaIm.retrieveImage(staticPrintF);
			overlapCudaIm.retrieveImage(overlapPrintf);

			writeImage(staticPrintF,staticCudaIm.getDimension(),"static_z%d.tif");
			writeImage(overlapPrintf,overlapCudaIm.getDimension(),"overlap_z%d.tif");


			staticCudaIm.maximumIntensityProjection();
			overlapCudaIm.maximumIntensityProjection();

			double staticSigma, overlapSigma;
			double curCorr = calcCorrelation(staticCudaIm,overlapCudaIm,staticSigma,overlapSigma);

			report[reportInd.x + reportInd.y*deltaSizes.x + reportInd.z*deltaSizes.y*deltaSizes.x].delta = Vec<int>(deltaX,deltaY,0);
			report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].correlation = curCorr;
			report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].staticSig = staticSigma;
			report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].overlapSig = overlapSigma;
			report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].nVoxels = szs.staticIm.x*szs.staticIm.y;

			if (curCorr>maxCorrelation)
			{
				maxCorrelation = curCorr;
				bestDelta.x = deltaX;
				bestDelta.y = deltaY;
				bestDelta.z = 0;
				bestN = szs.staticIm.x*szs.staticIm.y;
			}
			++curIter;
			//  				time(&zEnd);
			//  				zSec = difftime(zEnd,zStart);
			//  			time(&yEnd);
			//  			ySec = difftime(yEnd,yStart);
			//  			
			delete[] staticRoi;
			delete[] overlapRoi;
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

	printf("   (%d) Delta (%d,%d,%d) max:%f mipsTotalTime:%f avgTime:%f\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,maxCorrelation,mipSec,mipSec/iterations);

	iterations = deltaMaxs.z-deltaMins.z;
	reportInd.z = 0;
	//time(&yStart);
	starts.staticIm.x = (unsigned int)std::max<int>(0,overlap.deltaXss+bestDelta.x-imStarts.staticIm.x);
	starts.overlapIm.x = (unsigned int)std::max<int>(0,-(overlap.deltaXss+bestDelta.x)-imStarts.overlapIm.x);
	szs.staticIm.x = szs.overlapIm.x = (unsigned int)std::min<int>(imSizes.staticIm.x,overlap.deltaXse+bestDelta.x+1) - starts.staticIm.x;

	starts.staticIm.y = (unsigned int)std::max<int>(0,overlap.deltaYss+bestDelta.y-imStarts.staticIm.y);
	starts.overlapIm.y = (unsigned int)std::max<int>(0,-(overlap.deltaYss+bestDelta.y)-imStarts.overlapIm.y);
	szs.staticIm.y =szs.overlapIm.y = (unsigned int)std::min<int>(imSizes.staticIm.y,overlap.deltaYse+bestDelta.y+1) - starts.staticIm.y;

	for (int deltaZ=deltaMins.z; deltaZ<deltaMaxs.z; ++deltaZ, ++reportInd.z)
	{
		starts.staticIm.z = (unsigned int)std::max<int>(0,overlap.deltaZss+deltaZ-imStarts.staticIm.z);
		starts.overlapIm.z = (unsigned int)std::max<int>(0,-(overlap.deltaZss+deltaZ)-imStarts.overlapIm.z);
		szs.staticIm.z = szs.overlapIm.z = (unsigned int)std::min<int>(imSizes.staticIm.z,overlap.deltaZse+deltaZ+1) - starts.staticIm.z;

		const float* staticRoi = staticImage->getFloatConstROIData(starts.staticIm,szs.staticIm);
		const float* overlapRoi = overlapImage->getFloatConstROIData(starts.overlapIm,szs.overlapIm);

		CudaImageBuffer<float> staticCudaIm(szs.staticIm,deviceNum);
		CudaImageBuffer<float> overlapCudaIm(szs.overlapIm,deviceNum);

		staticCudaIm.loadImage(staticRoi);
		overlapCudaIm.loadImage(overlapRoi);

		double staticSigma, overlapSigma;
		double curCorr = calcCorrelation(staticCudaIm,overlapCudaIm,staticSigma,overlapSigma);

		report[reportInd.x + reportInd.y*deltaSizes.x + reportInd.z*deltaSizes.y*deltaSizes.x].delta = Vec<int>(bestDelta.x,bestDelta.y,deltaZ);
		report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].correlation = curCorr;
		report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].staticSig = staticSigma;
		report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].overlapSig = overlapSigma;
		report[reportInd.x+reportInd.y*deltaSizes.x+reportInd.z*deltaSizes.y*deltaSizes.x].nVoxels = (unsigned int)szs.staticIm.product();

		if (curCorr>maxCorrelation)
		{
			maxCorrelation = curCorr;
			bestDelta.z = deltaZ;
			bestN = (unsigned int)szs.staticIm.product();
		}
		++curIter;

		//time(&zStart);

		// 	int deltaX = 0;
		// 	int deltaY = 0;
		// 	int deltaZ = 0;
		// 	
		
		delete[] staticRoi;
		delete[] overlapRoi;

		if (0==deltaZ%5)
		{
			time(&xEnd);
			xSec = difftime(xEnd,xStart);

			printf("\t(%d)  BestCorr:%6.4f(%4d,%4d,%3d)",
				deviceNum, maxCorrelation, bestDelta.x, bestDelta.y, bestDelta.z);

			printf("  Done:%5.2f%% (%4d:%4d,%4d,%2d)",
				(float)curIter/iterations*100.0, bestDelta.x, overlap.deltaXmax, overlap.deltaYmax,overlap.deltaZmax);

			printf("  X:%6.2f avgY:%5.3f avgZ:%6.4f Est(min):%6.2f\n",
				xSec, xSec/(overlap.deltaYmax-overlap.deltaYmin),
				xSec/((overlap.deltaYmax-overlap.deltaYmin)*(overlap.deltaZmax-overlap.deltaZmin)),
				(iterations-curIter)*(xSec/((overlap.deltaYmax-overlap.deltaYmin)*(overlap.deltaZmax-overlap.deltaZmin)))/60.0);
		}
	}
	time(&mainEnd);
	mainSec = difftime(mainEnd,mainStart);

	offsets.x = ((deltaMaxs.x-deltaMins.x)/3.0)/2.0;
	offsets.y = ((deltaMaxs.y-deltaMins.y)/3.0)/2.0;
	offsets.z = ((deltaMaxs.z-deltaMins.z)/3.0)/2.0;

	deltaMins.x = std::max<int>(overlap.deltaXmin, (int)round((double)bestDelta.x - offsets.x));
	deltaMaxs.x = std::min<int>(overlap.deltaXmax, (int)round((double)bestDelta.x + offsets.x));
	deltaMins.y = std::max<int>(overlap.deltaYmin, (int)round((double)bestDelta.y - offsets.y));
	deltaMaxs.y = std::min<int>(overlap.deltaYmax, (int)round((double)bestDelta.y + offsets.y));
	deltaMins.z = std::max<int>(overlap.deltaZmin, (int)round((double)bestDelta.z - offsets.z));
	deltaMaxs.z = std::min<int>(overlap.deltaZmax, (int)round((double)bestDelta.z + offsets.z));

	printf("   (%d) Delta (%d,%d,%d) max:%f totalTime:%f\n",deviceNum,bestDelta.x,bestDelta.y,bestDelta.z,maxCorrelation,mainSec);
	FILE* reportFile;
	fopen_s(&reportFile,fileName,"w");
	for (int i=0; i<reportInd.product(); ++i)
	{
		fprintf(reportFile,"(%d,%d,%d):%lf,%lf,%lf,%d\n",report[i].delta.x,report[i].delta.y,report[i].delta.z,
			report[i].correlation,report[i].staticSig,report[i].overlapSig,report[i].nVoxels);
	}
	fclose(reportFile);

}
