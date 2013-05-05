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
			ov.deltaXse = ov.deltaXss + (int)gImageTiffs[overlapImageInd]->getXSize()-1;
			ov.deltaXmax = min(MARGIN, (int)gImageTiffs[staticImageInd]->getXSize() - MIN_OVERLAP - ov.deltaXss);
			ov.deltaXmin = max(-MARGIN, (MIN_OVERLAP-1) - ov.deltaXse);

			ov.deltaYss = gImageTiffs[overlapImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize() - gImageTiffs[staticImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize();
			ov.deltaYse = ov.deltaYss + (int)gImageTiffs[overlapImageInd]->getYSize()-1;
			ov.deltaYmax = min(MARGIN, (int)gImageTiffs[staticImageInd]->getYSize() - MIN_OVERLAP - ov.deltaYss);
			ov.deltaYmin = max(-MARGIN, (MIN_OVERLAP-1) - ov.deltaYse);

			ov.deltaZss = gImageTiffs[overlapImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize() - gImageTiffs[staticImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize();
			ov.deltaZse = ov.deltaZss + (int)gImageTiffs[overlapImageInd]->getZSize()-1;
			ov.deltaZmax = min(MARGIN, (int)gImageTiffs[staticImageInd]->getZSize() - MIN_OVERLAP_Z - ov.deltaZss);
			ov.deltaZmin = max(-MARGIN, (MIN_OVERLAP_Z-1) - ov.deltaZse);

			if (ov.deltaXmax-ov.deltaXmin>0 && ov.deltaYmax-ov.deltaYmin>0)
			{
				ov.ind = overlapImageInd;
				overlaps[staticImageInd].push_back(ov);
			}
		}
	}

	for (int staticImageInd=0; staticImageInd<overlaps.size(); ++staticImageInd)
	{
// #ifndef _DEBUG
// 		#pragma omp parallel for default(none) shared(overlaps,staticImageInd,gImageTiffs) num_threads(2)
// #endif
		for (int overlapImageInd=0; overlapImageInd<overlaps[staticImageInd].size(); ++overlapImageInd)
		{
			char buffer[255];
			sprintf(buffer,"%s_corrResults.txt",gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());

			//if(!gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->isAlligned() && !fileExists(buffer))
			//{
				Vec<int> deltas;
				double maxCorr;
//#ifndef _DEBUG
// 				printf("(%d) %s <-- %s\n",omp_get_thread_num(),gImageTiffs[staticImageInd]->getDatasetName().c_str(),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());
// 				ridgidRegistration(gImageTiffs[staticImageInd]->getImage(SCAN_CHANNEL,0),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getImage(SCAN_CHANNEL,0),overlaps[staticImageInd][overlapImageInd],deltas,maxCorr,omp_get_thread_num());
// #else
 				printf("(%d) %s <-- %s\n",1,gImageTiffs[staticImageInd]->getDatasetName().c_str(),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());
 				ridgidRegistration(gImageTiffs[staticImageInd]->getImage(SCAN_CHANNEL,0),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getImage(SCAN_CHANNEL,0),overlaps[staticImageInd][overlapImageInd],deltas,maxCorr,1);
// #endif
				edge curEdge;
				curEdge.deltas = deltas;
				curEdge.node1 = staticImageInd;
				curEdge.node2 = overlapImageInd;

				edgeList.insert(std::pair<double,edge>(-maxCorr,curEdge));

// 				deltas.x += gImageTiffs[staticImageInd]->getDeltas().x;
// 				deltas.y += gImageTiffs[staticImageInd]->getDeltas().y;
// 				deltas.z += gImageTiffs[staticImageInd]->getDeltas().z;
// 
// 				FILE* f = fopen(buffer,"wt");
// 				fprintf(f,"deltaX:%d\ndeltaY:%d\ndeltaZ:%d\nMaxCorr:%lf\n",deltas.x,deltas.y,deltas.z,maxCorr);
// 				fclose(f);

				//gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->setDeltas(deltas);
			//}
		}
	}

	while(visitedNodes.size()<gImageTiffs.size())
	{
		std::multimap<double,edge>::iterator best = edgeList.begin();
		if (visitedNodes.count(best->second.node1)>0 && visitedNodes.count(best->second.node1)>0)
		{
			edgeList.erase(best);
			continue;
		}

		visitedNodes.insert(best->second.node1);
		visitedNodes.insert(best->second.node2);
		bestEdges.push_back(*best);
		edgeList.erase(best);
	}


}