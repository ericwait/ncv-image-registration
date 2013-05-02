#ifndef RIDGED_REGISTRATION_H
#define RIDGED_REGISTRATION_H

#include "ImagesTiff.h"
#include "AlignImages.h"

void ridgedRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap, int& deltaX, int& deltaY, int deviceNum);
#endif