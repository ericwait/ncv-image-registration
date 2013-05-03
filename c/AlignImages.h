#ifndef ALIGN_IMAGES_H
#define ALIGN_IMAGES_H

#define MARGIN (75)
#define MIN_OVERLAP (25)
#define MIN_OVERLAP_Z (10)

struct Overlap
{
	int ind;

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

void align();
#endif