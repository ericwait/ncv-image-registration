#include "AlignImages.h"
#include "main.h"
#include "RidgidRegistration.h"
#include <omp.h>

//#define _USE_OMP

std::multimap<double,edge> edgeList;
std::set<int> visitedNodes;
std::vector<std::map<int,edge>> edges;
int scanChannel = 0;

//#pragma optimize("",off)

void addEdge(Vec<int> deltas, int curNode, int parentNode, double maxCorr)
{
	std::map<int,edge>::iterator it = edges[curNode].begin();
	for (; it!=edges[curNode].end(); ++it)
	{
			Vec<int> newDelta = deltas + it->second.deltas;
			addEdge(newDelta,it->second.node2,it->second.node1,it->second.maxCorr);
	}

	char buffer[255];
	sprintf(buffer,"%s_corrResults.txt",gImageTiffs[curNode]->getDatasetName().c_str());

	FILE* f = fopen(buffer,"wt");
	fprintf(f,"deltaX:%d\ndeltaY:%d\ndeltaZ:%d\nMaxCorr:%lf\nParent:%s\n",deltas.x,deltas.y,deltas.z,maxCorr,gImageTiffs[parentNode]->getDatasetName().c_str());
	fclose(f);
}

void align()
{
	//////////////////////////////////////////////////////////////////////////
	// Find which images overlap each other
	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<Overlap>> overlaps;
	overlaps.resize(gImageTiffs.size()-1);
	for (int staticImageInd = 0; staticImageInd < gImageTiffs.size()-1 ; staticImageInd++)
	{
		for (int overlapImageInd = staticImageInd+1; overlapImageInd < gImageTiffs.size() ; overlapImageInd++)
		{
			Overlap ov;
			ov.deltaXss = gImageTiffs[overlapImageInd]->getXPosition()/gImageTiffs[staticImageInd]->getXPixelPhysicalSize() - 
				gImageTiffs[staticImageInd]->getXPosition()/gImageTiffs[staticImageInd]->getXPixelPhysicalSize();

			ov.deltaXse = ov.deltaXss + (int)gImageTiffs[overlapImageInd]->getXSize()-1;
			ov.deltaXmax = min(MARGIN, (int)gImageTiffs[staticImageInd]->getXSize() - MIN_OVERLAP - ov.deltaXss);
			ov.deltaXmin = max(-MARGIN, (MIN_OVERLAP-1) - ov.deltaXse);

			ov.deltaYss = gImageTiffs[overlapImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize() - 
				gImageTiffs[staticImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize();

			ov.deltaYse = ov.deltaYss + (int)gImageTiffs[overlapImageInd]->getYSize()-1;
			ov.deltaYmax = min(MARGIN, (int)gImageTiffs[staticImageInd]->getYSize() - MIN_OVERLAP - ov.deltaYss);
			ov.deltaYmin = max(-MARGIN, (MIN_OVERLAP-1) - ov.deltaYse);

			ov.deltaZss = gImageTiffs[overlapImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize() - 
				gImageTiffs[staticImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize();

			ov.deltaZse = ov.deltaZss + (int)gImageTiffs[overlapImageInd]->getZSize()-1;
			ov.deltaZmax = min(MARGIN-50, (int)gImageTiffs[staticImageInd]->getZSize() - MIN_OVERLAP_Z - ov.deltaZss);
			ov.deltaZmin = max(-(MARGIN-50), (MIN_OVERLAP_Z-1) - ov.deltaZse);

			if (ov.deltaXmax-ov.deltaXmin>0 && ov.deltaYmax-ov.deltaYmin>0)
			{
				ov.ind = overlapImageInd;
				overlaps[staticImageInd].push_back(ov);
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Calculate the best overlap 
	//////////////////////////////////////////////////////////////////////////
	for (int staticImageInd=0; staticImageInd<overlaps.size(); ++staticImageInd)
	{
 #ifdef _USE_OMP
 		#pragma omp parallel for default(none) shared(overlaps,staticImageInd,gImageTiffs) num_threads(2)
 #endif
		for (int overlapImageInd=0; overlapImageInd<overlaps[staticImageInd].size(); ++overlapImageInd)
		{
				Vec<int> deltas;
				double maxCorr;
				unsigned int bestN;
				char buffer[255];
				sprintf_s(buffer,"%s_%s_report.txt",gImageTiffs[staticImageInd]->getDatasetName().c_str(),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());
#ifdef _USE_OMP
 				printf("(%d) %s <-- %s\n",omp_get_thread_num(),gImageTiffs[staticImageInd]->getDatasetName().c_str(),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());
 				ridgidRegistration(gImageTiffs[staticImageInd]->getImage(scanChannel,0),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getImage(scanChannel,0),overlaps[staticImageInd][overlapImageInd],deltas,maxCorr,omp_get_thread_num(),buffer);
 #else
 				printf("\n(%d) %s <-- %s\n",1,gImageTiffs[staticImageInd]->getDatasetName().c_str(),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());
 				ridgidRegistration(gImageTiffs[staticImageInd]->getImage(scanChannel,0),gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getImage(scanChannel,0),overlaps[staticImageInd][overlapImageInd],deltas,maxCorr,bestN,0,buffer);
 #endif
				edge curEdge;
				curEdge.deltas = deltas;
				curEdge.node1 = staticImageInd;
				curEdge.node2 = overlaps[staticImageInd][overlapImageInd].ind;
				curEdge.maxCorr = maxCorr;

				edgeList.insert(std::pair<double,edge>(-maxCorr*bestN,curEdge));
				curEdge.node1 = curEdge.node2;
				curEdge.node2 = staticImageInd;
				curEdge.deltas = -curEdge.deltas;
				edgeList.insert(std::pair<double,edge>(-maxCorr*bestN,curEdge));
		}
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Build the min span tree to make the registration calculations from
	//////////////////////////////////////////////////////////////////////////
	edges.resize(gImageTiffs.size());
	visitedNodes.insert(0);
	while(visitedNodes.size()<gImageTiffs.size())
	{
		std::multimap<double,edge>::iterator best = edgeList.begin();
		for (; best!=edgeList.end(); ++best)
		{
			if (visitedNodes.count(best->second.node1)==1 && visitedNodes.count(best->second.node2)==0)
			{
				visitedNodes.insert(best->second.node2);
				edges[best->second.node1].insert(std::pair<int,edge>(best->second.node2,best->second));
				edgeList.erase(best);
				break;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Travers tree to set final registration metadata
	//////////////////////////////////////////////////////////////////////////
	Vec<int> nullDeltas(0,0,0);
	addEdge(nullDeltas,0,0,0.0);
	//////////////////////////////////////////////////////////////////////////
}