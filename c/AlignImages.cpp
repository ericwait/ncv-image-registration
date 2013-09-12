#include "AlignImages.h"
#include "main.h"
#include "RidgidRegistration.h"
#include <omp.h>

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
	fprintf(f,"deltaX:%d\ndeltaY:%d\ndeltaZ:%d\nMaxCorr:%lf\nParent:%s\n",
		deltas.x, deltas.y, deltas.z, maxCorr, gImageTiffs[parentNode]->getDatasetName().c_str());
	fclose(f);
}

void align(std::string rootFolder, const int numberOfGPUs)
{
	//////////////////////////////////////////////////////////////////////////
	// Find which images overlap each other
	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<Overlap>> overlaps;
	overlaps.resize(gImageTiffs.size()-1);
	unsigned int totalEdges = 0;
	for (int staticImageInd = 0; staticImageInd < gImageTiffs.size()-1 ; staticImageInd++)
	{
		for (int overlapImageInd = staticImageInd+1; overlapImageInd < gImageTiffs.size() ; overlapImageInd++)
		{
			Overlap ov;
			ov.deltaXss = (int)(gImageTiffs[overlapImageInd]->getXPosition()/gImageTiffs[staticImageInd]->getXPixelPhysicalSize() - 
				gImageTiffs[staticImageInd]->getXPosition()/gImageTiffs[staticImageInd]->getXPixelPhysicalSize());

			ov.deltaXse = ov.deltaXss + (int)gImageTiffs[overlapImageInd]->getXSize()-1;
			ov.deltaXmax = std::min<int>(MARGIN, (int)gImageTiffs[staticImageInd]->getXSize() - MIN_OVERLAP - ov.deltaXss);
			ov.deltaXmin = std::max<int>(-MARGIN, (MIN_OVERLAP-1) - ov.deltaXse);

			ov.deltaYss = (int)(gImageTiffs[overlapImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize() - 
				gImageTiffs[staticImageInd]->getYPosition()/gImageTiffs[staticImageInd]->getYPixelPhysicalSize());

			ov.deltaYse = ov.deltaYss + (int)gImageTiffs[overlapImageInd]->getYSize()-1;
			ov.deltaYmax = std::min<int>(MARGIN, (int)gImageTiffs[staticImageInd]->getYSize() - MIN_OVERLAP - ov.deltaYss);
			ov.deltaYmin = std::max<int>(-MARGIN, (MIN_OVERLAP-1) - ov.deltaYse);

			ov.deltaZss = (int)(gImageTiffs[overlapImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize() - 
				gImageTiffs[staticImageInd]->getZPosition()/gImageTiffs[staticImageInd]->getZPixelPhysicalSize());

			ov.deltaZse = ov.deltaZss + (int)gImageTiffs[overlapImageInd]->getZSize()-1;
			ov.deltaZmax = std::min<int>(MARGIN, (int)gImageTiffs[staticImageInd]->getZSize() - MIN_OVERLAP_Z - ov.deltaZss);
			ov.deltaZmin = std::max<int>(-MARGIN, (MIN_OVERLAP_Z-1) - ov.deltaZse);

			if (ov.deltaXmax-ov.deltaXmin>0 && ov.deltaYmax-ov.deltaYmin>0)
			{
				ov.ind = overlapImageInd;
				overlaps[staticImageInd].push_back(ov);
				++totalEdges;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Calculate the best overlap 
	//////////////////////////////////////////////////////////////////////////
	unsigned int curEdgeNum = 0;
	for (int staticImageInd=0; staticImageInd<overlaps.size(); ++staticImageInd)
	{
 		#pragma omp parallel for default(none) shared(overlaps,staticImageInd,gImageTiffs,scanChannel,edgeList,totalEdges,curEdgeNum,rootFolder) num_threads(numberOfGPUs)
		for (int overlapImageInd=0; overlapImageInd<overlaps[staticImageInd].size(); ++overlapImageInd)
		{
			int deviceNum =0;
			deviceNum = omp_get_thread_num();
				Vec<int> deltas;
				double maxCorr;
				unsigned int bestN;
				char buffer[255];

				sprintf_s(buffer,"%s\\%s_%s_report.txt",rootFolder.c_str(),gImageTiffs[staticImageInd]->getDatasetName().c_str(),
					gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str());

 				printf("\n(%d) %s <-- %s Done: %3.1f%% %d of %d\n",deviceNum,gImageTiffs[staticImageInd]->getDatasetName().c_str(),
					gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getDatasetName().c_str(),
					(float)curEdgeNum/totalEdges*100.0,curEdgeNum,totalEdges);

 				ridgidRegistration(gImageTiffs[staticImageInd]->getImage(scanChannel,0),
					gImageTiffs[overlaps[staticImageInd][overlapImageInd].ind]->getImage(scanChannel,0),
					overlaps[staticImageInd][overlapImageInd],deltas,maxCorr,bestN,deviceNum,buffer);

				edge curEdge;
				curEdge.deltas = deltas;
				curEdge.node1 = staticImageInd;
				curEdge.node2 = overlaps[staticImageInd][overlapImageInd].ind;
				curEdge.maxCorr = maxCorr;

				#pragma omp critical
				{
					edgeList.insert(std::pair<double,edge>(-maxCorr*bestN,curEdge));
					curEdge.node1 = curEdge.node2;
					curEdge.node2 = staticImageInd;
					curEdge.deltas = -curEdge.deltas;
					edgeList.insert(std::pair<double,edge>(-maxCorr*bestN,curEdge));
					++curEdgeNum;
				}
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