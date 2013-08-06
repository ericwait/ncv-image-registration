#include "main.h"
#include "Utility.h"
#include "AlignImages.h"

std::vector<ImagesTiff*> gImageTiffs;

int main(int argc, char* argv[])
{
	std::string fileListLocation = "";

	HRESULT hr = S_FALSE;

	if (argc<3)
	{
		printf("Usage: %s listfile.txt channel\n",argv[0]);
		return 1;
	}

	fileListLocation = argv[1];
	scanChannel = atoi(argv[2])-1;
	

	if (!fileExists(fileListLocation.c_str()))
	{
		printf("%s does not exist!\n",fileListLocation.c_str());
		return 1;
	}

	std::vector<std::string> metadataFiles;
	std::ifstream file(fileListLocation.c_str());
	if (file.is_open())
	{
		while(file.good())
		{
			std::string line;
			getline(file,line);
			if (!line.empty())
			{
				metadataFiles.push_back(line);
			}
		}
		file.close();
	}else
	{
		printf("Cannot open %s!\n",fileListLocation.c_str());
		return 1;
	}

	size_t ind = fileListLocation.find_last_of("\\");
	std::string root = fileListLocation.substr(0,ind);
	//printf("%s\n",root.c_str());
	gImageTiffs.resize(metadataFiles.size());
	for (int i=0; i<metadataFiles.size(); ++i)
	{
		std::string mf = root;
		mf += "\\";
		mf += metadataFiles[i];
		mf += "\\";
		mf += metadataFiles[i];
		mf += ".txt";
		printf("%s\n",mf.c_str());
		ImagesTiff* im = new ImagesTiff(mf);
		gImageTiffs[i] = im;
	}

	align();

	for (int i = 0; i < gImageTiffs.size() ; i++)
		delete gImageTiffs[i];


	return 0;
}