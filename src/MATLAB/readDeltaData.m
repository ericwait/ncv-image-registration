function [deltasPresent,imageDatasets] = readDeltaData(root,imageDatasets)

if ~exist('root','var')
    rootDir = uigetdir('');
    if rootDir==0, return, end
else
    rootDir = root;
end
deltasPresent = 0;
names = {imageDatasets(:).DatasetName};
names = cellfun(@(x)([x '.']),names,'uniformOutput',0);
for i=1:length(imageDatasets)
    filename = fullfile(root,[imageDatasets(i).DatasetName '_corrResults.txt']);
   
    if exist(filename,'file') 
        deltasPresent = 1;
        fid = fopen(filename,'rt');
        data = fscanf(fid,'deltaX:%d\ndeltaY:%d\ndeltaZ:%d\nNCV:%f\n');
        deltaParent = [strtrim(fscanf(fid,'Parent:%255c\n')) '.'];
        fclose(fid);
        
        imageDatasets(i).xDelta = data(1);
        imageDatasets(i).yDelta = data(2);
        imageDatasets(i).zDelta = data(3);
        imageDatasets(i).NCV = data(4);
    else
        deltaParent = '';
        imageDatasets(i).xDelta = 0;
        imageDatasets(i).yDelta = 0;
        imageDatasets(i).zDelta = 0;
        imageDatasets(i).NCV = 0;
    end
    
    imageDatasets(i).ParentDelta = find(cellfun(@(x)(~isempty(x)),strfind(names,deltaParent)));
    if isempty(imageDatasets(i).ParentDelta)
        imageDatasets(i).ParentDelta = i;
    end
end

if (deltasPresent==1)
    imageDatasets(1).Children = [];
    root = [];
    for i=1:length(imageDatasets)
        if (imageDatasets(i).ParentDelta == i)
            root = [root,i];
            continue;
        end
        imageDatasets(imageDatasets(i).ParentDelta).Children = [imageDatasets(imageDatasets(i).ParentDelta).Children i];
    end
    
    if (length(root)~=1), error('There are too many roots to min span tree!'); end
    
    imageDatasets = applyParentsDelta(root,0,0,0,imageDatasets);
else
    for i=1:length(imageDatasets)
        imageDatasets(i).xMinPos = imageDatasets(i).YPosition*1e6;
        imageDatasets(i).xMaxPos = imageDatasets(i).xMinPos + imageDatasets(i).XDimension * imageDatasets(i).XPixelPhysicalSize;
        imageDatasets(i).yMinPos = imageDatasets(i).XPosition*1e6;
        imageDatasets(i).yMaxPos = imageDatasets(i).yMinPos + imageDatasets(i).YDimension * imageDatasets(i).YPixelPhysicalSize;
        imageDatasets(i).zMinPos = 0;
        imageDatasets(i).zMaxPos = imageDatasets(i).ZDimension * imageDatasets(i).ZPixelPhysicalSize;
    end
end
end

function imageDatasets = applyParentsDelta(root,deltaX,deltaY,deltaZ,imageDatasets)
imageDatasets(root).xDelta = imageDatasets(root).xDelta + deltaX;
imageDatasets(root).yDelta = imageDatasets(root).yDelta + deltaY;
imageDatasets(root).zDelta = imageDatasets(root).zDelta + deltaZ;

deltaX = imageDatasets(root).xDelta;
deltaY = imageDatasets(root).yDelta;
deltaZ = imageDatasets(root).zDelta;

imageDatasets(root).xMinPos = ...
    imageDatasets(root).xDelta * imageDatasets(root).XPixelPhysicalSize + imageDatasets(root).YPosition*1e6;

imageDatasets(root).xMaxPos = ...
    imageDatasets(root).xMinPos + imageDatasets(root).XDimension * imageDatasets(root).XPixelPhysicalSize;

imageDatasets(root).yMinPos = ...
    imageDatasets(root).yDelta * imageDatasets(root).YPixelPhysicalSize + imageDatasets(root).XPosition*1e6;

imageDatasets(root).yMaxPos = ...
    imageDatasets(root).yMinPos + imageDatasets(root).YDimension * imageDatasets(root).YPixelPhysicalSize;

imageDatasets(root).zMinPos = ...
    imageDatasets(root).zDelta * imageDatasets(root).ZPixelPhysicalSize;

imageDatasets(root).zMaxPos = ...
    imageDatasets(root).zMinPos + imageDatasets(root).ZDimension * imageDatasets(root).ZPixelPhysicalSize;

for i=1:length(imageDatasets(root).Children)
    imageDatasets = applyParentsDelta(imageDatasets(root).Children(i),deltaX,deltaY,deltaZ,imageDatasets);
end
end