#ifndef CUDA_KERNALS_H
#define CUDA_KERNALS_H

#include "cuda.h"
#include "ImagesTiff.h"

__global__ void meanFilter(float* imageIn, float* imageOut, int imageWidth, int imageHeight, int imageDepth, int kernalDiameter);
__global__ void addConstantInPlace(float* image, int width, int height, int depth, int additive);
__global__ void powerInPlace(float* image, int width, int height, int depth, int power);
__global__ void sqrtInPlace(float* image, int width, int height, int depth);
__global__ void multiplyTwoImages(const float* imageIn1, const float* imageIn2, float* imageOut, int width, int height, int depth);
__global__ void getROI(const float* imageIn, int orgSizeX, int orgSizeY, int orgSizeZ, float* imageOut, int startX, int startY, 
	int startZ, int newWidth, int newHeight, int newDepth);
__global__ void reduceArray(float* arrayIn, float* arrayOut, unsigned int n);
__global__ void reduceImage(PixelType* imageIn, PixelType* imageOut, unsigned int inWidth, unsigned int inHeight,
	unsigned int inDepth, unsigned int outWidth, unsigned int outHeight, unsigned int outDepth, unsigned int reduction);

#endif