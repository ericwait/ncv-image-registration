#ifndef ALIGN_IMAGES_H
#define ALIGN_IMAGES_H

#include <map>
#include <set>
#include <vector>
#include "Vec.h"

#define MARGIN (150)
#define LOCAL_REGION (15)
#define MIN_OVERLAP (25)
#define MIN_OVERLAP_Z (20)
extern int scanChannel;

struct Overlap
{
	int ind;
//////////////////////////////////////////////////////////////////////////
// Definitions:
// 
//  deltaSs:	Distance between the start position of the static image 
//				and the overlap image's starting position
//				
//  deltaSe:	Distance between the start position of the static image
//				and the overlap image's end position
//				
//	deltaMin:	Distance we are willing to move the start position of the 
//				overlap image to the left 
//	
//	deltaMax:	The distance we are will to move the start position of the
//				overlap image to the right
//////////////////////////////////////////////////////////////////////////
	int deltaXss;
	int deltaXse;
	int deltaXmax;
	int deltaXmin;

	int deltaYss;
	int deltaYse;
	int deltaYmax;
	int deltaYmin;

	int deltaZss;
	int deltaZse;
	int deltaZmax;
	int deltaZmin;
};

template<typename T>
struct comparedImages
{
	Vec<T> staticIm;
	Vec<T> overlapIm;
};

struct edge
{
	Vec<int> deltas;
	int node1;
	int node2;
	double maxCorr;
};

void align(std::string rootFolder, const int numberOfGPUs=1);
#endif