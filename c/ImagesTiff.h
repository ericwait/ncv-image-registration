#ifndef IMAGES_TIFF_H
#define IMAGES_TIFF_H

#include "winHead.h"
#include "Utility.h"
#include "AlignImages.h"

#define MAX_CHANNELS (6) //Make sure that this is synced with the .fx file
#define DIMENSION (3)

typedef unsigned char PixelType; 

bool fileExists(const char* filename);
DWORD WINAPI processingThread(LPVOID lpParam);
void writeImage(const PixelType* image, unsigned int width, unsigned int height, unsigned int depth, std::string fileName);
void writeImage(const float* image, unsigned int width, unsigned int height, unsigned int depth, std::string fileName);

class ImageContainer
{
public:
	ImageContainer(unsigned int width, unsigned int height, unsigned int depth);
	ImageContainer(const ImageContainer& image){copy(image);}
	~ImageContainer(){clear();}
	ImageContainer& operator=(const ImageContainer& image){copy(image);}

	std::string getName() const {return name;}
	PixelType getPixelValue(unsigned int x, unsigned int y, unsigned int z) const;
	const PixelType* getConstMemoryPointer() const {return image;}
	const PixelType* getConstROIData (unsigned int minX, unsigned int maxX, unsigned int minY, unsigned int maxY, unsigned int minZ, unsigned int sizeZ) const;
	const float* getConstFloatROIData (unsigned int minX, unsigned int maxX, unsigned int minY, unsigned int maxY, unsigned int minZ, unsigned int sizeZ) const;
	PixelType* getMemoryPointer(){return image;}
	unsigned int getWidth() const {return width;}
	unsigned int getHeight() const {return height;}
	unsigned int getDepth() const {return depth;}
	float getXPosition() const {return xPosition;}
	float getYPosition() const {return yPosition;}
	float getZPosition() const {return zPosition;}

	void setPixelValue(unsigned int x, unsigned int y, unsigned int z,PixelType val);
	void setExtents(unsigned int width, unsigned int height, unsigned int depth);
	void setWidth(unsigned int width){this->width=width;}
	void setHeight(unsigned int height){this->height=height;}
	void setDepth(unsigned int depth){this->depth=depth;}
	void setName(std::string name){this->name=name;}

private:
	ImageContainer();
	void copy(const ImageContainer& image);
	void clear();
	void defaults() 
	{
		name		= "";
		width		= -1;
		height		= -1;
		depth		= -1;
		xPosition	= 0.0;
		yPosition	= 0.0;
		zPosition	= 0.0;
	}

	std::string		name;
	unsigned int	width;
	unsigned int	height;
	unsigned int	depth;
	double			xPosition;
	double			yPosition;
	double			zPosition;

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
	unsigned int getXSize() const {return xSize;}
	unsigned int getYSize() const {return ySize;}
	unsigned int getZSize() const {return zSize;}
	float getXScale() const {return xScale;}
	float getYScale() const {return yScale;}
	float getZScale() const {return zScale;}
	float getXPixelPhysicalSize() const {return xPixelPhysicalSize;}
	float getYPixelPhysicalSize() const {return yPixelPhysicalSize;}
	float getZPixelPhysicalSize() const {return zPixelPhysicalSize;}
	float getXPosition() const {return xPosition;}
	float getYPosition() const {return yPosition;}
	float getZPosition() const {return zPosition;}
	PixelType getPixel(unsigned char channel, unsigned int frame, unsigned int x, unsigned int y, unsigned int z) const;
	bool isAlligned(){return alligned;}
	Vec<int> getDeltas(){return deltas;}

	//Setters
	void setImage(ImageContainer& image, unsigned char channel, unsigned int frame);
	//void resetImagesToOrg(const unsigned char CHANNEL);
	void setDatasetName(std::string datasetName){this->datasetName=datasetName;}
	void setNumberOfChannels(unsigned char numberOfChannels){this->numberOfChannels=numberOfChannels;}
	void setNumberOfFrames(unsigned int numberOfFrames){this->numberOfFrames=numberOfFrames;}
	void setXPixelPhysicalSize(float xPixelPhysicalSize){this->xPixelPhysicalSize=xPixelPhysicalSize;}
	void setYPixelPhysicalSize(float yPixelPhysicalSize){this->yPixelPhysicalSize=yPixelPhysicalSize;}
	void setZPixelPhysicalSize(float zPixelPhysicalSize){this->zPixelPhysicalSize=zPixelPhysicalSize;}
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
	int				xSize;
	int				ySize;
	int				zSize;
	double			xPixelPhysicalSize;
	double			yPixelPhysicalSize;
	double			zPixelPhysicalSize;
	double			xScale;
	double			yScale;
	double			zScale;
	double			xPosition;
	double			yPosition;
	double			zPosition;

	unsigned int	maxThreads;
	bool			alligned;
	Vec<int>		deltas;
};//*gImagesTiff;

#endif