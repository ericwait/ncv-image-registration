#ifndef ALIGN_IMAGES_H
#define ALIGN_IMAGES_H

#define MARGIN (150)

struct Overlap
{
	int ind;
	int staticXminInd;
	int staticYminInd;
	int staticZminInd;

	int overlapXminInd;
	int overlapYminInd;
	int overlapZminInd;

	int xSize;
	int ySize;
	int zSize;
};

void align();
#endif