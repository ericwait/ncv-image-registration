#include "ImagesTiff.h"
#include <limits.h>
#include "AlignImages.h"

#include "tiffio.h"
#include "tiff.h"

#define MAX_THREAD_USAGE (0.5)
#define CToMat(x) ((x)+1)

//extern CRITICAL_SECTION gProcessingCritical;
CRITICAL_SECTION gSegmentationCritical;

bool fileExists(const char* filename){
	std::ifstream ifile(filename);
	bool rtn = ifile.good();
	ifile.close();
	return rtn;
}

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

DWORD WINAPI processingThread(LPVOID lpParam)
{
	//EnterCriticalSection(&gProcessingCritical);

	//char buffer[255];
	//paramPassing* param = (paramPassing*)lpParam;
	//if (param->functionToCall.compare("backgroundSubstract")==0)
	//{
	//	sprintf_s(buffer," *Background Subtraction running on channel:%d",CToMat(param->channel));
	//	gWindowSuffix = buffer;
	//	updateWindowTitle("");
	//	gImagesTiff->backgroundSubtraction(param->channel);
	//}
	//else if (param->functionToCall.compare("MFRdenoise")==0)
	//{
	//	sprintf_s(buffer," *MRF Denoise running on channel:%d",CToMat(param->channel));
	//	gWindowSuffix = buffer;
	//	updateWindowTitle("");
	//	gImagesTiff->MRFdenoise(param->channel);
	//}
	//else if (param->functionToCall.compare("medianFilter")==0)
	//{
	//	sprintf_s(buffer," *Median Denoise running on channel:%d",CToMat(param->channel));
	//	gWindowSuffix = buffer;
	//	updateWindowTitle("");
	//	gImagesTiff->medianFilter(param->channel);
	//}
	//else if (param->functionToCall.compare("resetChannel")==0)
	//{
	//	gImagesTiff->resetImagesToOrg(param->channel);
	//}
	//else if (param->functionToCall.compare("Segment")==0)
	//{
	//	InitializeCriticalSection(&gSegmentationCritical);
	//	sprintf_s(buffer," *Segmenting channel:%d",CToMat(param->channel));
	//	gWindowSuffix = buffer;
	//	updateWindowTitle("");
	//	gImagesTiff->segment(param->channel);
	//	DeleteCriticalSection(&gSegmentationCritical);

	//	sprintf_s(buffer," *Tracking channel:%d",CToMat(param->channel));
	//	gWindowSuffix = buffer;
	//	updateWindowTitle("");

	//	trackHulls();

	//	for (int i=0; i<gHulls.size(); ++i)
	//	{
	//		gHulls[i].initBuffers();
	//	}

	//	g_bRender = false;
	//	GetTrackLists();

	//	char buffer[255];
	//	sprintf_s(buffer,".\\data\\%s\\%s.lvr",gImagesTiff->getDatasetName(),gImagesTiff->getDatasetName());
	//	gSavePath = buffer;
	//	SaveData();

	//	gSegment = true;
	//	g_drawHulls = true;
	//	gAllHulls = true;

	//	g_bRender = true;
	//}

	//delete param;

// 	g_state = REFRESH;
// 	gWindowSuffix = "";
// 
// 	updateWindowTitle("");

	//LeaveCriticalSection(&gProcessingCritical);

	return S_OK;
}

//////////////////////////////////////////////////////////////////////////
//ImageBuffer
//////////////////////////////////////////////////////////////////////////
ImageContainer::ImageContainer(unsigned int width, unsigned int height, unsigned int depth)
{
	defaults();
	this->width = width;
	this->height = height;
	this->depth = depth;

	image = new PixelType[width*height*depth];
}

void ImageContainer::copy(const ImageContainer& im)
{
	clear();
	name = im.getName();
	width = im.getWidth();
	height = im.getHeight();
	depth = im.getDepth();
	xPosition = im.getXPosition();
	yPosition = im.getYPosition();
	zPosition = im.getZPosition();

	image = new PixelType[width*height*depth];
	memcpy((void*)image,(void*)(im.getConstMemoryPointer()),sizeof(PixelType)*width*height*depth);
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
#ifdef _DEBUG
	assert(width!=-1 && x<width);
	assert(height!=-1 && y<height);
	assert(depth!=-1 && z<depth);
#endif
	return image[x+y*width+z*height*width];
}

void ImageContainer::setPixelValue(unsigned int x, unsigned int y, unsigned int z, unsigned char val)
{
#ifdef _DEBUG
	assert(width!=-1 && x<width);
	assert(height!=-1 && y<height);
	assert(depth!=-1 && z<depth);
#endif
	if (x>width || y>height || z>depth)
		return;

	image[x+y*width+z*height*width] = val;
}

const PixelType* ImageContainer::getConstROIData (unsigned int minX, unsigned int sizeX, unsigned int minY, unsigned int sizeY, unsigned int minZ, unsigned int sizeZ) const
{
	assert(sizeX<=width);
	assert(sizeY<=height);
	assert(sizeZ<=depth);

	PixelType* image = new PixelType[sizeX*sizeY*sizeZ];

	unsigned int i=0;
	for (unsigned int z=0; z<sizeZ; ++z)
		for (unsigned int y=minY; y<sizeY; ++y)
			for (unsigned int x=minX; x<sizeX+1; ++x)		
				image[i] = (float)getPixelValue(x,y,z);

	return image;
}

const float* ImageContainer::getConstFloatROIData (unsigned int minX, unsigned int sizeX, unsigned int minY, unsigned int sizeY, unsigned int minZ, unsigned int sizeZ) const
{
	assert(sizeX<=minX+width);
	assert(sizeY<=minY+height);
	assert(sizeZ<=minZ+depth);

	float* image = new float[sizeX*sizeY*sizeZ];

	unsigned int i=0;
	for (unsigned int z=minZ; z<minZ+sizeZ; ++z)
	{
		for (unsigned int y=minY; y<minY+sizeY; ++y)
		{
			for (unsigned int x=minX; x<minX+sizeX; ++x)		
			{
				image[i] = (float)getPixelValue(x,y,z);
				++i;
			}
		}
	}
	if (i<sizeX*sizeY*sizeZ)
		int m = sizeX*sizeY*sizeZ;

	return image;
}

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//ImagesTiff
//////////////////////////////////////////////////////////////////////////
ImagesTiff::ImagesTiff(const std::string metaDataFile)
{
// 	if (gImagesTiff==NULL)
// 		gImagesTiff = this;

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
	xSize = 0.0f;
	ySize = 0.0f;
	zSize = 0.0f;
	xPixelPhysicalSize = 1.0;
	yPixelPhysicalSize = 1.0;
	zPixelPhysicalSize = 1.0;
	xPosition = 0.0;
	yPosition = 0.0;
	zPosition = 0.0;
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
		this->xSize = atoi(metadata["XDimension"].c_str());

	if (metadata.count("YDimension")!=0)
		this->ySize = atoi(metadata["YDimension"].c_str());

	if (metadata.count("ZDimension")!=0)
		this->zSize = atoi(metadata["ZDimension"].c_str());

	if (metadata.count("XPixelPhysicalSize")!=0)
		this->setXPixelPhysicalSize(atof(metadata["XPixelPhysicalSize"].c_str()));

	if (metadata.count("YPixelPhysicalSize")!=0)
		this->setYPixelPhysicalSize(atof(metadata["YPixelPhysicalSize"].c_str()));

	if (metadata.count("ZPixelPhysicalSize")!=0)
		this->setZPixelPhysicalSize(atof(metadata["ZPixelPhysicalSize"].c_str()));

	if (metadata.count("XPixelPhysicalSize")!=0)
		this->setXPixelPhysicalSize(atof(metadata["XPixelPhysicalSize"].c_str()));

	if (metadata.count("XPosition")!=0)
		this->yPosition = atof(metadata["XPosition"].c_str()) * 1e6;

	if (metadata.count("YPosition")!=0)
		this->xPosition = atof(metadata["YPosition"].c_str()) * 1e6;

	setScales();

	// TODO: Get all the times
}

void ImagesTiff::setScales()
{
	float maxDim = std::max(std::max(xSize,ySize),zSize);
	xScale = xSize/maxDim;
	yScale = ySize/maxDim * (yPixelPhysicalSize/xPixelPhysicalSize);
	zScale = zSize/maxDim * (zPixelPhysicalSize/xPixelPhysicalSize);
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
	this->xPixelPhysicalSize = 0.0;
	this->yPixelPhysicalSize = 0.0;
	this->zPixelPhysicalSize = 0.0;
	xPosition = 0.0;
	yPosition = 0.0;
	zPosition = 0.0;
	this->alligned = false;
	deltas.x = 0;
	deltas.y = 0;
	deltas.z = 0;
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
#ifdef _DEBUG
	assert(channel<numberOfChannels);
	assert(frame<numberOfFrames);
#endif
	return imageBuffers[channel][frame]->getPixelValue(x,y,z);
}

void ImagesTiff::segment(const unsigned char CHANNEL)
{
	//if (CHANNEL>=numberOfChannels)
	//	return;

	//gHashedHulls.resize(numberOfFrames);
	//int nThreads = maxThreads*MAX_THREAD_USAGE;

	//int numFrames = numberOfFrames;
	//++numberOfChannels;
	//imagesChar.resize(numberOfChannels);
	//imagesChar[numberOfChannels-1].resize(numberOfFrames);

	//#pragma omp parallel for default(none) shared(numFrames,CHANNEL) num_threads(nThreads)
	//for (int frame=0; frame<numFrames; ++frame)
	//{
	//	SegmentFrame(frame,CHANNEL);
	//}

	//system("del outfile*.txt errfile*.txt && exit");
}

void ImagesTiff::medianFilter(const unsigned char CHANNEL)
{
	//if (CHANNEL>=numberOfChannels || CHANNEL<0)
	//	return;

	//int nThreads = maxThreads*MAX_THREAD_USAGE;
	//#pragma omp parallel for default(none) shared(CHANNEL) num_threads(nThreads)
	//for (int frame=0; frame<numberOfFrames; ++frame)
	//{
	//	CharImageVolumeType::Pointer newImage = medianDenoise(this->getImage(CHANNEL,frame));
	//	this->setImage(newImage,CHANNEL,frame);
	//}
}

void ImagesTiff::backgroundSubtraction(const unsigned char CHANNEL)
{
	//if (CHANNEL>=numberOfChannels || CHANNEL<0)
	//	return;

	//int nThreads = maxThreads*MAX_THREAD_USAGE;
	//#pragma omp parallel for default(none) shared(CHANNEL) num_threads(nThreads)
	//for (int frame=0; frame<numberOfFrames; ++frame)
	//{
	//	CharImageVolumeType::Pointer newImage = subtractBackground(this->getImage(CHANNEL,frame));
	//	this->setImage(newImage,CHANNEL,frame);
	//}
}
//
//void ImagesTiff::MRFdenoise(const unsigned char CHANNEL)
//{
//	if (CHANNEL>=numberOfChannels || CHANNEL<0)
//		return;
//
//	int nThreads = maxThreads*MAX_THREAD_USAGE;
//	#pragma omp parallel for default(none) shared(CHANNEL) num_threads(nThreads)
//	for (int frame=0; frame<numberOfFrames; ++frame)
//	{
//		CharImageVolumeType::Pointer newImage = markovDenoise(this->getImage(CHANNEL,frame));
//		this->setImage(newImage,CHANNEL,frame);
//	}
//}

//void ImagesTiff::resetImagesToOrg(const unsigned char CHANNEL)
//{
//	//if (CHANNEL>=numberOfChannels || CHANNEL<0)
//	//	return;
//
//	//for (unsigned int frame=0; frame<numberOfFrames; ++frame)
//	//	imagesChar[CHANNEL][frame] = imagesCharOrg[CHANNEL][frame];
//
//	//g_state = REFRESH;
//}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//ITK
//////////////////////////////////////////////////////////////////////////
void ImagesTiff::setupCharReader()
{
#ifdef _DEBUG
	assert(this->getNumberOfChannels()>0);
	assert(this->getNumberOfFrames()>0);
	assert(this->xPixelPhysicalSize>0);
	assert(this->yPixelPhysicalSize>0);
	assert(this->zPixelPhysicalSize>0);
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
	unsigned int stripCount=0, stripSize=0, imageOffset=0, result=0, width=0, height=0, depth=0;
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

	imageBuffers[channel][frame]->setName(this->datasetName);
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
				image[x+y*width+z*height*width] = (PixelType)std::max(std::min(floatImage[x+y*width+z*height*width],255.0f),0.0f);
				
			}
		}
	}

	writeImage(image,width,height,depth,fileName);

	delete image;
}

void writeImage(const PixelType* image, unsigned int width, unsigned int height, unsigned int depth, std::string fileName)
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