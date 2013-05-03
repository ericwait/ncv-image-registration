#ifndef RIDGID_REGISTRATION_H
#define RIDGID_REGISTRATION_H

#include "ImagesTiff.h"
#include "AlignImages.h"

void ridgidRegistration(const ImageContainer* staticImage, const ImageContainer* overlapImage, const Overlap& overlap, int& deltaX, int& deltaY, int& deltaZ, int deviceNum);
#endif