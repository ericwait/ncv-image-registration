#ifndef IMAGES_TIFF_H
#define IMAGES_TIFF_H

#include "winHead.h"
#include "Utility.h"
#include "AlignImages.h"
#include "Vec.h"

#define MAX_CHANNELS (6) //Make sure that this is synced with the .fx file
#define DIMENSION (3)

typedef unsigned char PixelType; 

bool fileExists(const char* filename);
DWORD WINAPI processingThread(LPVOID lpParam);

/*
 *	This will write out a series of tif images where the passed in prefix is used to name the z stack written.
 *	Use printf syntax for this string.
 *	e.g. "image_z%d" will give you a image image_z1.tif or "image_z%04d" will give you image_z0001.tif
 */
void writeImage(const PixelType* image, unsigned int width, unsigned int height, unsigned int depth, std::string fileNamePrefix);

/*
 *	This will write out a series of tif images where the passed in prefix is used to name the z stack written.
 *	Use printf syntax for this string.
 *	e.g. "image_z%d" will give you a image image_z1.tif or "image_z%04d" will give you image_z0001.tif
 */
void writeImage(const float* image, unsigned int width, unsigned int height, unsigned int depth, std::string fileNamePrefix);

/*
 *	This will write out a series of tif images where the passed in prefix is used to name the z stack written.
 *	Use printf syntax for this string.
 *	e.g. "image_z%d" will give you a image image_z1.tif or "image_z%04d" will give you image_z0001.tif
 */
void writeImage(const PixelType* imageBuffer, Vec<unsigned int> dims, std::string fileNamePrefix);

/*
 *	This will write out a series of tif images where the passed in prefix is used to name the z stack written.
 *	Use printf syntax for this string.
 *	e.g. "image_z%d" will give you a image image_z1.tif or "image_z%04d" will give you image_z0001.tif
 */
void writeImage(const float* floatImage, Vec<unsigned int> dims, std::string fileNamePrefix);

class ImageContainer
{
public:
	ImageContainer(unsigned int width, unsigned int height, unsigned int depth);
	ImageContainer(Vec<unsigned int> dims);
	ImageContainer(const ImageContainer& image){copy(image);}
	~ImageContainer(){clear();}
	ImageContainer& operator=(const ImageContainer& image){copy(image); return *this;}

	std::string getName() const {return name;}
	PixelType getPixelValue(unsigned int x, unsigned int y, unsigned int z) const;
	PixelType getPixelValue(Vec<unsigned int> coordinate) const;
	const PixelType* getConstMemoryPointer() const {return image;}
	const PixelType* ImageContainer::getConstROIData (unsigned int minX, unsigned int sizeX, unsigned int minY,
		unsigned int sizeY, unsigned int minZ, unsigned int sizeZ) const;
	const PixelType* getConstROIData(Vec<unsigned int> startIndex, Vec<unsigned int> size) const;
	const float* getFloatConstROIData(Vec<unsigned int> startIndex, Vec<unsigned int> size) const;
	const double* getDoubleConstROIData(Vec<unsigned int> startIndex, Vec<unsigned int> size) const;
	PixelType* getMemoryPointer(){return image;}
	Vec<unsigned int> getDims() const {return dims;}
	unsigned int getWidth() const {return dims.x;}
	unsigned int getHeight() const {return dims.y;}
	unsigned int getDepth() const {return dims.z;}
	Vec<double> getPositions() const {return positions;}
	double getXPosition() const {return positions.x;}
	double getYPosition() const {return positions.y;}
	double getZPosition() const {return positions.z;}

	void setPixelValue(unsigned int x, unsigned int y, unsigned int z, unsigned char val);
	void setPixelValue(Vec<unsigned int> coordinate,PixelType val);
	void setExtents(Vec<unsigned int> dims){this->dims=dims;}
	void setWidth(unsigned int width){dims.x=width;}
	void setHeight(unsigned int height){dims.y=height;}
	void setDepth(unsigned int depth){dims.z=depth;}
	void setName(std::string name){this->name=name;}

private:
	ImageContainer();
	void copy(const ImageContainer& image);
	void clear();
	void defaults() 
	{
		name		= "";
		dims = Vec<unsigned int>((unsigned int)-1,(unsigned int)-1,(unsigned int)-1);
		positions = Vec<double>(0.0,0.0,0.0);
	}

	std::string		name;
	Vec<unsigned int> dims;
	Vec<double> positions;

	PixelType*	image;
};

void writeImage(const ImageContainer* image, std::string fileName);

class ImagesTiff
{
public:
	ImagesTiff(const std::string imagesFolder);
	~ImagesTiff(){clear();}

	//Getters
	ImageContainer* getImage(unsigned char channel, unsigned int frame);
	const PixelType* getConstImageData(unsigned char channel, unsigned int frame) const;
	std::string	getDatasetName() const {return datasetName;}
	std::string	getImagesPath() const {return imagesPath;}
	unsigned char getNumberOfChannels() const {return numberOfChannels;}
	unsigned int getNumberOfFrames() const {return numberOfFrames;}
	Vec<unsigned long long> getSizes() const {return sizes;}
	unsigned long long  getXSize() const {return sizes.x;}
	unsigned long long  getYSize() const {return sizes.y;}
	unsigned long long  getZSize() const {return sizes.z;}
	Vec<double> getScales() const {return scales;}
	double getXScale() const {return scales.x;}
	double getYScale() const {return scales.y;}
	double getZScale() const {return scales.z;}
	Vec<double> getPixelPhysicalSizes() const {return pixelPhysicalSizes;}
	double getXPixelPhysicalSize() const {return pixelPhysicalSizes.x;}
	double getYPixelPhysicalSize() const {return pixelPhysicalSizes.y;}
	double getZPixelPhysicalSize() const {return pixelPhysicalSizes.z;}
	Vec<double> getPositions() const {return positions;}
	double getXPosition() const {return positions.x;}
	double getYPosition() const {return positions.y;}
	double getZPosition() const {return positions.z;}
	PixelType getPixel(unsigned char channel, unsigned int frame, unsigned int x, unsigned int y, unsigned int z) const;
	PixelType getPixel(unsigned char channel, unsigned int frame, Vec<unsigned int> coordinate) const;
	bool isAlligned(){return alligned;}
	Vec<int> getDeltas(){return deltas;}

	//Setters
	void setImage(ImageContainer& image, unsigned char channel, unsigned int frame);
	//void resetImagesToOrg(const unsigned char CHANNEL);
	void setDatasetName(std::string datasetName){this->datasetName=datasetName;}
	void setNumberOfChannels(unsigned char numberOfChannels){this->numberOfChannels=numberOfChannels;}
	void setNumberOfFrames(unsigned int numberOfFrames){this->numberOfFrames=numberOfFrames;}
	void setPixelPhysicalSizes(Vec<double> pixelPhysicalSizes){this->pixelPhysicalSizes=pixelPhysicalSizes;}
	void setXPixelPhysicalSize(double xPixelPhysicalSize){this->pixelPhysicalSizes.x=xPixelPhysicalSize;}
	void setYPixelPhysicalSize(double yPixelPhysicalSize){this->pixelPhysicalSizes.y=yPixelPhysicalSize;}
	void setZPixelPhysicalSize(double zPixelPhysicalSize){this->pixelPhysicalSizes.z=zPixelPhysicalSize;}
	void setAlligned(bool isAlligned){this->alligned = isAlligned;}
	void setDeltas(Vec<int> bestDeltas){deltas = bestDeltas;}

	//Processors
	void segment(const unsigned char CHANNEL);
	void medianFilter(const unsigned char CHANNEL);
	void backgroundSubtraction(const unsigned char CHANNEL);
	//void MRFdenoise(const unsigned char CHANNEL);

private:
	void clear();
	void reset();
	void sizeImageBuffer();
	void emptyImageBuffers();
	bool readMetadata(std::string metadataFile);
	void setMetadata(std::map<std::string,std::string> metadata);
	void setScales();
	void setupCharReader();
	void reader(unsigned char channel, unsigned int frame);

	void setImagesPath(std::string imagesPath){this->imagesPath=imagesPath;}

	std::vector<std::vector<ImageContainer*>> imageBuffers;
	std::string		datasetName;
	std::string		imagesPath;
	unsigned char	numberOfChannels;
	unsigned short	numberOfFrames;
	float			timeBetweenFrames;
	Vec<unsigned long long> sizes;
	Vec<double>		pixelPhysicalSizes;
	Vec<double>		scales;
	Vec<double>		positions;

	unsigned int	maxThreads;
	bool			alligned;
	Vec<int>		deltas;
};//*gImagesTiff;

#endif