#include "RidgidRegistration.h"
#include "CudaImageBuffer.h"

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
	Vec<int>& bestDelta, double& maxCorrOut, unsigned int& bestN, int deviceNum, const char* filename)
{
	comparedImages<unsigned int> imStarts, imSizes;
	Vec<int> maxOverlapSize;
	Vec<int> staticImageExtents(staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth());

	calcMaxROIs(overlap,staticImageExtents,imStarts,imSizes,maxOverlapSize);

	const float* staticMaxRoi = staticImage->getFloatConstROIData(imStarts.staticIm,imSizes.staticIm);
	const float* overlapMaxRoi = overlapImage->getFloatConstROIData(imStarts.staticIm,imSizes.staticIm);

	CudaImageBuffer<float> staticCudaIm(imSizes.staticIm,deviceNum);
	CudaImageBuffer<float> overlabCudaIm(imSizes.overlapIm,deviceNum);

}
