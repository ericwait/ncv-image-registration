function im = tiffReader(type,chanList,timeList,zList,path,metadataFile)
global imageData

if (~exist('path','var') || isempty(path))
    [metadataFile, path] = uigetfile('.txt');
end

fileHandle = fopen(fullfile(path,metadataFile),'r');
if fileHandle<=0
    error('Cannot Open Metadata');
end

data = textscan(fileHandle,'%s', 'delimiter',':','whitespace','\n');
fclose(fileHandle);
if isempty(data)
    error('File malformed');
end

imageData.DatasetName = data{1}{2};
imageData.NumberOfChannels = str2double(data{1}{4});
imageData.NumberOfFrames = str2double(data{1}{6});
imageData.xDim = str2double(data{1}{8});
imageData.yDim = str2double(data{1}{10});
imageData.zDim = str2double(data{1}{12});
imageData.xVoxelSize = str2double(data{1}{14});
imageData.yVoxelSize = str2double(data{1}{16});
imageData.zVoxelSize = str2double(data{1}{18});

if (~exist('type','var') || isempty(type))
    bytes=8;
elseif (strcmp(type,'uint8'))
    bytes=1;
else
    error('Unsupported Type');
end

if (~exist('chanList','var') || isempty(chanList))
    chanList = 1:imageData.NumberOfChannels;
end
if (~exist('timeList','var') || isempty(timeList))
    timeList = 1:imageData.NumberOfFrames;
end
if (~exist('zList','var') || isempty(zList))
    zList = 1:imageData.zDim;
end

if (bytes==1)
    im = zeros(imageData.yDim,imageData.xDim,length(zList),length(chanList),length(timeList),'uint8');
elseif (bytes==8)
    im = zeros(imageData.yDim,imageData.xDim,length(zList),length(chanList),length(timeList));
end

fprintf('(');
fprintf('%d',size(im,2));
fprintf(',%d',size(im,1));
for i=3:length(size(im))
    fprintf(',%d',size(im,i));
end

fprintf(') %5.2fMB\n', (imageData.xDim*imageData.yDim*length(zList)*length(chanList)*length(timeList)*bytes)/(1024*1024));

for c=1:length(chanList)
    for t=1:length(timeList)
        for z=1:length(zList)
            if (bytes==1)
                im(:,:,z,t,c) = uint8(imread(fullfile(path,sprintf('%s_c%d_t%04d_z%04d.tif',imageData.DatasetName,chanList(c),timeList(t),zList(z))),'tif'));
            elseif (bytes==8)
                im(:,:,z,t,c) = imread(fullfile(path,sprintf('%s_c%d_t%04d_z%04d.tif',imageData.DatasetName,chanList(c),timeList(t),zList(z)),'tif'));
            end
        end
    end
end
end
