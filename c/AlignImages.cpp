#include "AlignImages.h"
#include "main.h"
#include "RidgedRegistration.h"

void align()
{
	// Find which images overlap each other
	std::vector<std::vector<Overlap>> overlaps;
	overlaps.resize(gImageTiffs.size()-1);
	for (int staticImageInd = 0; staticImageInd < gImageTiffs.size()-1 ; staticImageInd++)
	{
		for (int overlapImageInd = staticImageInd+1; overlapImageInd < gImageTiffs.size() ; overlapImageInd++)
		{
			int deltaX = (int)((gImageTiffs[staticImageInd]->getXPosition() - gImageTiffs[overlapImageInd]->getXPosition()) / gImageTiffs[staticImageInd]->getXPixelPhysicalSize());
			int deltaY = (int)((gImageTiffs[staticImageInd]->getYPosition() - gImageTiffs[overlapImageInd]->getYPosition()) / gImageTiffs[staticImageInd]->getYPixelPhysicalSize());

			if ((unsigned)std::abs(deltaX)<gImageTiffs[staticImageInd]->getXSize() && (unsigned)std::abs(deltaY)<gImageTiffs[staticImageInd]->getYSize())
			{
				Overlap ov;
				ov.ind = overlapImageInd;

				if (deltaX>0)
				{
					deltaX = max(deltaX-MARGIN, 0);
					ov.staticXminInd = deltaX;
					ov.xSize = min(gImageTiffs[staticImageInd]->getXSize(),gImageTiffs[overlapImageInd]->getXSize()) - deltaX;
					ov.overlapXminInd = 0;
				} 
				else if (deltaX<0)
				{
					deltaX = min(deltaX-MARGIN, -1) * -1;
					ov.staticXminInd = 0;
					ov.xSize = min(gImageTiffs[staticImageInd]->getXSize(),gImageTiffs[overlapImageInd]->getXSize()-deltaX);
					ov.overlapXminInd = gImageTiffs[overlapImageInd]->getXSize() - ov.xSize;
				}
				else
				{
					ov.staticXminInd = 0;
					ov.xSize = min(gImageTiffs[staticImageInd]->getXSize(),gImageTiffs[overlapImageInd]->getXSize());
					ov.overlapXminInd = 0;
				}

				if (deltaY>0)
				{
					deltaY = max(deltaY-MARGIN, 0);
					ov.staticYminInd = deltaY;
					ov.ySize = min(gImageTiffs[staticImageInd]->getYSize(),gImageTiffs[overlapImageInd]->getYSize());
					ov.overlapYminInd = 0;
				} 
				else if (deltaY<0)
				{
					deltaY = min(deltaY-MARGIN, -1) * -1;
					ov.staticYminInd = 0;
					ov.ySize = min(gImageTiffs[staticImageInd]->getYSize(),gImageTiffs[overlapImageInd]->getYSize()-deltaY);
					ov.overlapYminInd = gImageTiffs[overlapImageInd]->getYSize() - ov.ySize;
				}
				else
				{
					ov.staticYminInd = 0;
					ov.ySize = min(gImageTiffs[staticImageInd]->getYSize(),gImageTiffs[overlapImageInd]->getYSize());
					ov.overlapYminInd = 0;
				}

				ov.staticZminInd = 0;
				ov.overlapZminInd = 0;
				ov.zSize = min(gImageTiffs[staticImageInd]->getZSize(),gImageTiffs[overlapImageInd]->getZSize());

				overlaps[staticImageInd].push_back(ov);
			}
		}
	}

	for (int staticImageInd=0; staticImageInd<overlaps.size(); ++staticImageInd)
	{
		for (int overlapImageInd=0; overlapImageInd<overlaps[staticImageInd].size(); ++overlapImageInd)
		{
			for (int chan=3; chan<gImageTiffs[staticImageInd]->getNumberOfChannels(); ++chan)
			{
				int deltaX, deltaY;
				ridgedRegistration(gImageTiffs[staticImageInd]->getImage(chan,0),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getImage(chan,0),overlaps[staticImageInd][overlapImageInd],deltaX,deltaY,1);
			}
		}
	}

}