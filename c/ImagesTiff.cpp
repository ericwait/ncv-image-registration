#include "ImagesTiff.h"
#include <limits.h>
#include "AlignImages.h"

#include "tiffio.h"
#include "tiff.h"

#define CToMat(x) ((x)+1)

//extern CRITICAL_SECTION gProcessingCritical;
//CRITICAL_SECTION gSegmentationCritical;


LPCWSTR s2lp (const std::string& s)
{
	std::wstring ws;
	LPCWSTR ls;
	ws.assign (s.begin (), s.end ());
	ls = ws.c_str();
	return ls;
}

void updateWindowTitle(std::string msg)
{
	printf("%s\n",msg.c_str());
}

//////////////////////////////////////////////////////////////////////////
//ImageBuffer
//////////////////////////////////////////////////////////////////////////
ImageContainer::ImageContainer(unsigned int width, unsigned int height, unsigned int depth)
{
	defaults();
	dims.x = width;
	dims.y = height;
	dims.z = depth;

	image = new PixelType[dims.product()];
}

ImageContainer::ImageContainer(Vec<unsigned int> dimsIn)
{
	defaults();
	dims = dimsIn;

	image = new PixelType[dims.product()];
}

void ImageContainer::copy(const ImageContainer& im)
{
	clear();
	name = im.getName();
	dims = im.getDims();
	positions = im.getPositions();

	image = new PixelType[dims.product()];
	memcpy((void*)image,(void*)(im.getConstMemoryPointer()),sizeof(PixelType)*dims.product());
}

void ImageContainer::clear()
{
	defaults();

	if (image)
	{
		delete[] image;
		image = NULL;
	}
}

PixelType ImageContainer::getPixelValue(unsigned int x, unsigned int y, unsigned int z) const
{
	return getPixelValue(Vec<unsigned int>(x,y,z));
}

PixelType ImageContainer::getPixelValue(Vec<unsigned int> coordinate) const
{
#ifdef _DEBUG
	assert(width!=-1 && x<width);
	assert(height!=-1 && y<height);
	assert(depth!=-1 && z<depth);
#endif
	return image[dims.linearAddressAt(coordinate)];
}

void ImageContainer::setPixelValue(unsigned int x, unsigned int y, unsigned int z, unsigned char val)
{
	setPixelValue(Vec<unsigned int>(x,y,z),val);
}

void ImageContainer::setPixelValue(Vec<unsigned int> coordinate, unsigned char val)
{
#ifdef _DEBUG
	assert(coordinate<Vec<unsigned int>(-1,-1,-1));
#endif
	assert(coordinate<dims);

	image[dims.at(coordinate)] = val;
}

const PixelType* ImageContainer::getConstROIData (unsigned int minX, unsigned int sizeX, unsigned int minY,
	unsigned int sizeY, unsigned int minZ, unsigned int sizeZ) const
{
	return getConstROIData<PixelType>(Vec<unsigned int>(minX,minY,minZ), Vec<unsigned int>(sizeX,sizeY,sizeZ));
}

template<typename ImagePixelType>
const ImagePixelType* ImageContainer::getConstROIData (Vec<unsigned int> startIndex, Vec<unsigned int> size) const
{
	assert(size-startIndex<dims);

	PixelType* image = new PixelType[size.product()];

	unsigned int i=0;
	for (unsigned int z=startIndex.z; z<size.z; ++z)
		for (unsigned int y=startIndex.y; y<size.y; ++y)
			for (unsigned int x=startIndex.x; x<size.x+1; ++x)		
				image[i] = (ImagePixelType*)getPixelValue(x,y,z);

	return image;
}

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//ImagesTiff
//////////////////////////////////////////////////////////////////////////
ImagesTiff::ImagesTiff(const std::string metaDataFile)
{
	size_t separator = 0;
	clear();

	if (!readMetadata(metaDataFile))
	{
		std::cerr << "Unable to read: " << metaDataFile << std::endl;
		return;
	}

	std::string imPath;
	separator = metaDataFile.find_last_of("\\");
	imPath = metaDataFile.substr(0,separator);

	setImagesPath(imPath);

	setupCharReader();

	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );
	maxThreads = sysinfo.dwNumberOfProcessors;
}

void ImagesTiff::reset()
{
	datasetName = "";
	numberOfChannels = 1;
	numberOfFrames = 1;
	timeBetweenFrames = 1;
	sizes = Vec<unsigned long long>(0,0,0);
	pixelPhysicalSizes = Vec<double>(1.0,1.0,1.0);
	positions = Vec<double>(0.0,0.0,0.0);
	alligned = false;
}

bool ImagesTiff::readMetadata(std::string metadataFile)
{
	size_t separator = 0;
	if(!fileExists(metadataFile.c_str()))
		return false;
	std::map<std::string,std::string> metaData;
	std::ifstream file(metadataFile.c_str());
	if (file.is_open())
	{
		while(file.good())
		{
			std::string line;
			getline(file,line);
			if (!line.empty())
			{
				separator = line.find(":");
				std::string key = line.substr(0,separator);
				std::string val	= line.substr(separator+1,line.size()-separator);

				//Remove whitespace
				separator = key.find_first_of("\t");
				if (separator!=std::string::npos)
					key = key.substr(separator,key.length()-separator);

				separator = key.find_first_of(" ");
				while (separator!=std::string::npos)
				{
					key = key.substr(separator,key.length()-separator);
					separator = key.find_first_of(" ");
				}

				metaData.insert(std::pair<std::string,std::string>(key,val));
			}
		}
		file.close();
	} 
	else
		return false;
	
	setMetadata(metaData);
	sizeImageBuffer();

// 	char buffer[255];
// 	sprintf_s(buffer,".\\data\\%s\\%s.lvr",datasetName,datasetName);
// 	if (fileExists(buffer))
// 	{
// 		gSavePath = buffer;
// 		if(OpenData())
// 		{
// 			gSegment = true;
// 			g_drawHulls = true;
// 			gAllHulls = true;
// 		}
// 	}

	return true;
}

void ImagesTiff::sizeImageBuffer()
{
	if (!imageBuffers.empty())
		emptyImageBuffers();

	imageBuffers.resize(numberOfChannels);

	for (int chan=0; chan<numberOfChannels; ++chan)
		imageBuffers[chan].resize(numberOfFrames);
}

void ImagesTiff::setMetadata(std::map<std::string,std::string> metadata)
{
	if (metadata.count("DatasetName")!=0)
		this->setDatasetName(metadata["DatasetName"]);

	if (metadata.count("NumberOfChannels")!=0)
		this->setNumberOfChannels(atoi(metadata["NumberOfChannels"].c_str()));

	if (metadata.count("NumberOfFrames")!=0)
		this->setNumberOfFrames(atoi(metadata["NumberOfFrames"].c_str()));

	if (metadata.count("XDimension")!=0)
		this->sizes.x = atoi(metadata["XDimension"].c_str());

	if (metadata.count("YDimension")!=0)
		this->sizes.y = atoi(metadata["YDimension"].c_str());

	if (metadata.count("ZDimension")!=0)
		this->sizes.z = atoi(metadata["ZDimension"].c_str());

	if (metadata.count("XPixelPhysicalSize")!=0)
		this->setXPixelPhysicalSize(atof(metadata["XPixelPhysicalSize"].c_str()));

	if (metadata.count("YPixelPhysicalSize")!=0)
		this->setYPixelPhysicalSize(atof(metadata["YPixelPhysicalSize"].c_str()));

	if (metadata.count("ZPixelPhysicalSize")!=0)
		this->setZPixelPhysicalSize(atof(metadata["ZPixelPhysicalSize"].c_str()));

	if (metadata.count("XPixelPhysicalSize")!=0)
		this->setXPixelPhysicalSize(atof(metadata["XPixelPhysicalSize"].c_str()));

	if (metadata.count("XPosition")!=0)
		this->positions.x = atof(metadata["XPosition"].c_str()) * 1e6;

	if (metadata.count("YPosition")!=0)
		this->positions.y = atof(metadata["YPosition"].c_str()) * 1e6;

	if (metadata.count("ZPosition")!=0)
		this->positions.z = atof(metadata["ZPosition"].c_str()) * 1e6;

	setScales();

	// TODO: Get all the times
}

void ImagesTiff::setScales()
{
	scales.x = sizes.x/sizes.max();
	scales.y = sizes.y/sizes.max() * (pixelPhysicalSizes.y/pixelPhysicalSizes.x);
	scales.z = sizes.z/sizes.max() * (pixelPhysicalSizes.z/pixelPhysicalSizes.x);
}

ImageContainer* ImagesTiff::getImage(unsigned char channel, unsigned int frame)
{
#ifdef _DEBUG
	assert(channel<numberOfChannels);
	assert(frame<numberOfFrames);
#endif

	return imageBuffers[channel][frame];
}

const PixelType* ImagesTiff::getConstImageData(unsigned char channel, unsigned int frame) const
{
#ifdef _DEBUG
	assert(channel<numberOfChannels);
	assert(frame<numberOfFrames);
#endif
	return imageBuffers[channel][frame]->getConstMemoryPointer();
}

void ImagesTiff::emptyImageBuffers() 
{
	for (unsigned char chan=0; chan<numberOfChannels && chan<imageBuffers.size(); ++chan)
	{
		for (unsigned int frame=0; frame<numberOfFrames && frame<imageBuffers[chan].size(); ++frame)
			delete imageBuffers[chan][frame];

		imageBuffers[chan].clear();
	}

	imageBuffers.clear();
}

void ImagesTiff::clear ()
{
	emptyImageBuffers();

	this->datasetName = "";
	this->numberOfChannels = 0;
	this->numberOfFrames = 0;
	this->timeBetweenFrames = 0.0;
	this->pixelPhysicalSizes = Vec<double>(0.0,0.0,0.0);
	this->positions = Vec<double>(0.0,0.0,0.0);
	this->alligned = false;
	this->deltas = Vec<int>(0,0,0);
}

void ImagesTiff::setImage(ImageContainer& image, unsigned char channel, unsigned int frame)
{
#ifdef _DEBUG
	assert(channel<numberOfChannels);
	assert(frame<numberOfFrames);
#endif

	//Explicit removal of image smart pointer in the attempts to have ITK clean up
	delete imageBuffers[channel][frame];
	imageBuffers[channel][frame] = new ImageContainer(image);
}

PixelType ImagesTiff::getPixel(unsigned char channel, unsigned int frame, unsigned int x, unsigned int y, unsigned int z) const
{
	return getPixel(channel,frame,Vec<unsigned int>(x,y,z));
}

PixelType ImagesTiff::getPixel(unsigned char channel, unsigned int frame, Vec<unsigned int> coordinate) const
{
#ifdef _DEBUG
	assert(channel<numberOfChannels);
	assert(frame<numberOfFrames);
#endif
	return imageBuffers[channel][frame]->getPixelValue(coordinate);
}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//Image I/O
//////////////////////////////////////////////////////////////////////////
void ImagesTiff::setupCharReader()
{
#ifdef _DEBUG
	assert(this->getNumberOfChannels()>0);
	assert(this->getNumberOfFrames()>0);
	assert(pixelPhysicalSizes>Vec<double>(0.0,0.0,0.0));
#endif
	char buffer[255];

	for (int frame=0; frame<this->getNumberOfFrames(); ++frame)
	{
// 		for (int chan=0; chan<this->getNumberOfChannels(); ++chan)
// 		{
		int chan = scanChannel;
			sprintf_s(buffer,"Reading Image Data for Frame:%d/%d Channel:%d/%d",CToMat(frame),numberOfFrames,CToMat(chan),numberOfChannels);
			updateWindowTitle(buffer);
			reader(chan, frame);
		//} 
	}
	updateWindowTitle("");
}

void ImagesTiff::reader(unsigned char channel, unsigned int frame) 
{
	char filenameTemplate[255], curFileName[255];
	sprintf_s(filenameTemplate,"%s\\%s_c%d_t%04d_z%s.tif",imagesPath.c_str(),datasetName.c_str(),CToMat(channel),CToMat(frame),"%04d");
	printf("Reading:%s...\n",filenameTemplate);
	TIFF* image;
	unsigned int stripCount=0, stripSize=0, imageOffset=0, result=0, width=0, height=0, depth=this->zSize;
	unsigned short bps, spp;
	PixelType* imageBuffer;

	for (unsigned int z=0; z<depth; ++z)
	{
		imageOffset=0;
		sprintf_s(curFileName,filenameTemplate,z+1);
		if ((image=TIFFOpen(curFileName,"r"))==NULL)
		{
			fprintf(stderr,"Could not open %s\n",curFileName);
			continue;
		}

		// Check that it is of a type that we support
		if((TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bps) == 0) || (bps != 8)){
			fprintf(stderr, "Either undefined or unsupported number of bits per sample\n");
		}
		if((TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &spp) == 0) || (spp != 1)){
			fprintf(stderr, "Either undefined or unsupported number of samples per pixel\n");
			exit(42);
		}
		if (stripSize==0 && stripCount==0)
		{
			stripSize = TIFFStripSize (image);
			stripCount = TIFFNumberOfStrips (image);
			TIFFGetField(image,TIFFTAG_IMAGEWIDTH,&width);
			TIFFGetField(image,TIFFTAG_IMAGELENGTH,&height);
			imageBuffers[channel][frame] = new ImageContainer(width,height,depth);
			imageBuffer = imageBuffers[channel][frame]->getMemoryPointer();
		}
		else if (stripSize!=TIFFStripSize(image) || stripCount!=TIFFNumberOfStrips(image))
		{
			fprintf(stderr,"Image %s does not have the same dimension (%d,%d)!=(%d,%d)\n",curFileName,TIFFStripSize(image),
				TIFFNumberOfStrips(image),stripSize,stripCount);
			if(imageBuffer!=NULL)
			{
				delete imageBuffer;
				imageBuffer = NULL;
			}
		}
		for (unsigned int y=0; y<stripCount; ++y)
		{
			if ((result=TIFFReadEncodedStrip(image,y,imageBuffer+imageOffset+z*width*height,stripSize))==-1)
			{
				fprintf(stderr,"Read error on input strip number %d on image %s\n",y,curFileName);
				delete imageBuffer;
				imageBuffer = NULL;
			}

			imageOffset += result;
		}
		TIFFClose(image);
	}

	imageBuffers[channel][frame]->setName(this->datasetName.c_str());
}

void writeImage(const ImageContainer* image, std::string fileName)
{
	writeImage(image->getConstMemoryPointer(),image->getWidth(),image->getHeight(),image->getDepth(),fileName);
}

void writeImage(const float* floatImage, unsigned int width, unsigned int height, unsigned int depth, std::string fileName)
{
	PixelType* image = new PixelType[width*height*depth];

	for (unsigned int z=0; z<depth; ++z)
	{
		for (unsigned int y=0; y<height; ++y)
		{
			for (unsigned int x=0; x<width; ++x)
			{
				image[x+y*width+z*height*width] = (PixelType)MAX(MAX(floatImage[x+y*width+z*height*width],255.0f),0.0f);
				
			}
		}
	}

	writeImage(image,width,height,depth,fileName);

	delete image;
}

void writeImage(const PixelType* imageBuffer, unsigned int width, unsigned int height, unsigned int depth, std::string fileName)
{
	char curFileName[255];
	TIFF* image;

	printf("Writing:%s\n",fileName.c_str());
	for (unsigned int z=0; z<depth; ++z)
	{
		sprintf_s(curFileName,fileName.c_str(),z+1);
		// Open the TIFF file
		if((image = TIFFOpen(curFileName, "w")) == NULL){
			printf("Could not open %s for writing\n",curFileName);
			continue;
		}

		// We need to set some values for basic tags before we can add any data
		TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
		TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
		TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 8);
		TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, height);

		TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
		TIFFSetField(image, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
		TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

		// Write the information to the file
		TIFFWriteEncodedStrip(image, 0, (void*)(imageBuffer+z*width*height), width*height);

		// Close the file
		TIFFClose(image);
	}
}
//////////////////////////////////////////////////////////////////////////