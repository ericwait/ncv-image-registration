#include "AlignImages.h"
#include "main.h"
#include "RidgidRegistration.h"
#include <omp.h>


void align()
{
	// Find which images overlap each other
	std::vector<std::vector<Overlap>> overlaps;
	overlaps.resize(gImageTiffs.size()-1);
	for (int staticImageInd = 0; staticImageInd < gImageTiffs.size()-1 ; staticImageInd++)
	{
		for (int overlapImageInd = staticImageInd+1; overlapImageInd < gImageTiffs.size() ; overlapImageInd++)
		{
			Overlap ov;
			ov.deltaXss = gImageTiffs[overlapImageInd]->getXPosition()/gImageTiffs[staticImageInd]->getXPixelPhysicalSize() - gImageTiffs[staticImageInd]->getXPosition()/gImageTiffs[staticImageInd]->getXPixelPhysicalSize();
			ov.deltaXse = ov.deltaXss + gImageTiffs[overlapImageInd]->getXSize();
			ov.deltaXmax = min(MARGIN, gImageTiffs[staticImageInd]->getXSize() - MIN_OVERLAP - ov.deltaXss);
			ov.deltaXmin = max(-MARGIN, MIN_OVERLAP - ov.deltaXse);

			ov.deltaYss = gImageTiffs[overlapImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize() - gImageTiffs[staticImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize();
			ov.deltaYse = ov.deltaYss + gImageTiffs[overlapImageInd]->getYSize();
			ov.deltaYmax = min(MARGIN, gImageTiffs[staticImageInd]->getYSize() - MIN_OVERLAP - ov.deltaYss);
			ov.deltaYmin = max(-MARGIN, MIN_OVERLAP - ov.deltaYse);

			ov.deltaZss = gImageTiffs[overlapImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize() - gImageTiffs[staticImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize();
			ov.deltaZse = ov.deltaZss + gImageTiffs[overlapImageInd]->getZSize();
			ov.deltaZmax = min(MARGIN, gImageTiffs[staticImageInd]->getZSize() - MIN_OVERLAP_Z - ov.deltaZss);
			ov.deltaZmin = max(-MARGIN, MIN_OVERLAP_Z - ov.deltaZse);

			if (ov.deltaXmax-ov.deltaXmin<=0 && ov.deltaYmax-ov.deltaYmin<=0) // && all dims
				continue;

			ov.ind = overlapImageInd;
			overlaps[staticImageInd].push_back(ov);
		}
	}

	for (int staticImageInd=0; staticImageInd<overlaps.size(); ++staticImageInd)
	{
		#pragma omp parallel for default(none) shared(overlaps,staticImageInd,gImageTiffs) num_threads(2)
		for (int overlapImageInd=0; overlapImageInd<overlaps[staticImageInd].size(); ++overlapImageInd)
		{
				int deltaX, deltaY, deltaZ;
				printf("%s <-- %s\n",gImageTiffs[staticImageInd]->getDatasetName().c_str(),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());
				ridgidRegistration(gImageTiffs[staticImageInd]->getImage(3,0),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getImage(3,0),overlaps[staticImageInd][overlapImageInd],deltaX,deltaY,deltaZ,omp_get_thread_num());
		}
	}

}