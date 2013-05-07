#ifndef ALIGN_IMAGES_H
#define ALIGN_IMAGES_H

#include <map>
#include <set>
#include <vector>

#define MARGIN (150)
#define MIN_OVERLAP (25)
#define MIN_OVERLAP_Z (10)
#define SCAN_CHANNEL (2)

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
struct Vec
{
	T x;
	T y;
	T z;
	
	Vec()
	{
		x=0;
		y=0;
		z=0;
	}

	Vec(T x, T y, T z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vec<T> operator+ (Vec<T> other) const
	{
		Vec<T> outVec;
		outVec.x = x + other.x;
		outVec.y = y + other.y;
		outVec.z = z + other.z;

		return outVec;
	}

	Vec<T> operator- (Vec<T> other) const
	{
		Vec<T> outVec;
		outVec.x = x - other.x;
		outVec.y = y - other.y;
		outVec.z = z - other.z;

		return outVec;
	}

	Vec<T> operator- () const
	{
		Vec<T> outVec;
		outVec.x = -x;
		outVec.y = -y;
		outVec.z = -z;

		return outVec;
	}
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

void align();
#endif