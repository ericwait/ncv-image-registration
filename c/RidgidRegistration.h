#ifndef RIDGID_REGISTRATION_H
#define RIDGID_REGISTRATION_H

#include "ImagesTiff.h"
#include "AlignImages.h"
#include "Vec.h"

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap,
	Vec<int>& bestDelta, float& maxCorrOut, unsigned int& bestN, int deviceNum, const char* filename);
#endif