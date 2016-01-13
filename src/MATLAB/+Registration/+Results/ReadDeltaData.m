function [deltasPresent,imageDatasets] = ReadDeltaData(root,imageDatasets,unitFactor)
if (~exist('unitFactor','var') || isempty(unitFactor))
    unitFactor = 1e6;
end

if ~exist('root','var')
    rootDir = uigetdir('');
    if rootDir==0, return, end
else
    rootDir = root;
end
deltasPresent = false;
names = {imageDatasets(:).DatasetName};
names = cellfun(@(x)([x '.']),names,'uniformOutput',0);
for i=1:length(imageDatasets)
    filename = fullfile(root,imageDatasets(i).DatasetName,[imageDatasets(i).DatasetName '_corrResults.txt']);
   
    if exist(filename,'file') 
        deltasPresent = true;
        fid = fopen(filename,'rt');
        data = fscanf(fid,'deltaX:%d\ndeltaY:%d\ndeltaZ:%d\nNCV:%f\n');
        deltaParent = [strtrim(fscanf(fid,'Parent:%255c\n')) '.'];
        fclose(fid);
        
        imageDatasets(i).Delta(1) = data(1);
        imageDatasets(i).Delta(2) = data(2);
        imageDatasets(i).Delta(3) = data(3);
        imageDatasets(i).NCV = data(4);
    else
        deltaParent = '';
        imageDatasets(i).Delta(1) = 0;
        imageDatasets(i).Delta(2) = 0;
        imageDatasets(i).Delta(3) = 0;
        imageDatasets(i).NCV = 0;
    end
    
    imageDatasets(i).ParentDelta = find(cellfun(@(x)(~isempty(x)),strfind(names,deltaParent)));
    if isempty(imageDatasets(i).ParentDelta)
        imageDatasets(i).ParentDelta = i;
    end
end

if (deltasPresent==true)
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
    
    imageDatasets = applyParentsDelta(root,[0,0,0],imageDatasets,unitFactor);
else
    for i=1:length(imageDatasets)
        imageDatasets(i).MinPos = round(imageDatasets(i).Position*unitFactor .* imageDatasets(i).PixelPhysicalSize);
        imageDatasets(i).MaxPos = imageDatasets(i).MinPos + imageDatasets(i).Dimensions -1;
    end
end
end

function imageDatasets = applyParentsDelta(root,delta,imageDatasets,unitFactor)
imageDatasets(root).Delta = imageDatasets(root).Delta + delta;

delta = imageDatasets(root).Delta;

imageDatasets(root).MinPos = round(imageDatasets(root).Position*unitFactor .* imageDatasets(root).PixelPhysicalSize) + delta;
imageDatasets(root).MaxPos = imageDatasets(root).MinPos + imageDatasets(root).Dimensions -1;

for i=1:length(imageDatasets(root).Children)
    imageDatasets = applyParentsDelta(imageDatasets(root).Children(i),delta,imageDatasets,unitFactor);
end
end