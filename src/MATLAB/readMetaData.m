function readMetaData(root)
global imageDatasets rootDir datasetName

imageDatasets = [];

if ~exist('root','var')
    rootDir = uigetdir('');
    if rootDir==0,
        return
    end
else
    rootDir = root;
end

dlist = dir(fullfile(rootDir,[datasetName '*']));

for i=1:length(dlist)
    if (strcmp(dlist(i).name,'.') || strcmp(dlist(i).name,'..')),continue,end
    
    if (~isempty(strfind(dlist(i).name,'_Montage')))
        continue
    end
    
    dSublist = dir(fullfile(rootDir,dlist(i).name,'*.txt'));
    
    if isempty(dSublist)
        continue
    end
    
    for j=1:length(dSublist)
        corr = strfind(dlist(i).name,'_corrResults.txt');
        if ~isempty(corr)
            continue;
        end
        rep = strfind(dlist(i).name,'_report.txt');
        if ~isempty(rep)
            continue;
        end
        
        fileHandle = fopen(fullfile(rootDir,dlist(i).name,dSublist(j).name));
        if fileHandle<=0, continue, end
        data = textscan(fileHandle,'%s', 'delimiter',':','whitespace','\n');
        fclose(fileHandle);
        
        if isempty(data), continue, end
        
        imageDatum.DatasetName = data{1}{2};
        imageDatum.NumberOfChannels = str2double(data{1}{4});
        imageDatum.NumberOfFrames = str2double(data{1}{6});
        imageDatum.xDim = str2double(data{1}{8});
        imageDatum.yDim = str2double(data{1}{10});
        imageDatum.zDim = str2double(data{1}{12});
        imageDatum.xVoxelSize = str2double(data{1}{14});
        imageDatum.yVoxelSize = str2double(data{1}{16});
        imageDatum.zVoxelSize = str2double(data{1}{18});
        imageDatum.xMinPos = str2double(data{1}{20})*1e6;
        imageDatum.yMinPos = str2double(data{1}{22})*1e6;
        imageDatum.xMaxPos = imageDatum.xMinPos+imageDatum.xVoxelSize*imageDatum.xDim;
        imageDatum.yMaxPos = imageDatum.yMinPos+imageDatum.yVoxelSize*imageDatum.yDim;
        imageDatum.zMinPos = 0;
        imageDatum.zMaxPos = imageDatum.zVoxelSize*imageDatum.zDim;
        imageDatum.xDelta = 0;
        imageDatum.yDelta = 0;
        imageDatum.zDelta = 0;
        imageDatum.ParentDelta = 1;
        imageDatum.Child = [];
        if isempty(imageDatasets)
            imageDatasets = imageDatum;
        else        
            imageDatasets(length(imageDatasets)+1) = imageDatum;
        end
    end
end
end