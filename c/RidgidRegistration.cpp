#include "RidgidRegistration.h"
#include "CudaImageBuffer.h"

void calcMaxROIs(const Overlap& overlap, Vec<int> imageExtents, comparedImages<int>& imStarts,
	comparedImages<int>& imSizes, Vec<int>& maxOverlapSize)
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

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap,
	Vec<int>& bestDelta, double& maxCorrOut, unsigned int& bestN, int deviceNum, const char* filename)
{
	comparedImages<int> imStarts, imSizes;
	Vec<int> maxOverlapSize;
	Vec<int> staticImageExtents(staticImage->getWidth(),staticImage->getHeight(),staticImage->getDepth());

	calcMaxROIs(overlap,staticImageExtents,imStarts,imSizes,maxOverlapSize);


	const float* staticImageFloat = staticImage->getConstFloatROIData(imStarts.staticIm.x,imSizes.staticIm.x,
		imStarts.staticIm.y,imSizes.staticIm.y,imStarts.staticIm.z,imSizes.staticIm.z); 
	
	const float* overlapImageFloat = overlapImage->getConstFloatROIData(imStarts.overlapIm.x,imSizes.overlapIm.x,
		imStarts.overlapIm.y,imSizes.overlapIm.y,imStarts.overlapIm.z,imSizes.overlapIm.z);


}