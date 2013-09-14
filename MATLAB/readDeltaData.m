function readDeltaData(root)
global imageDatasets DeltasPresent

if ~exist('root','var')
    rootDir = uigetdir('');
    if rootDir==0, return, end
else
    rootDir = root;
end
DeltasPresent = 0;
names = {imageDatasets(:).DatasetName};
names = cellfun(@(x)([x '.']),names,'uniformOutput',0);
for i=1:length(imageDatasets)
    filename = fullfile(root,[imageDatasets(i).DatasetName '_corrResults.txt']);
   
    if ~exist(filename,'file')
        continue
    end

    DeltasPresent = 1;
    fid = fopen(filename,'rt');
    data = fscanf(fid,'deltaX:%d\ndeltaY:%d\ndeltaZ:%d\nMaxCorr:%f\n');
    deltaParent = [strtrim(fscanf(fid,'Parent:%255c\n')) '.'];
    fclose(fid);
    
    imageDatasets(i).xDelta = data(2);
    imageDatasets(i).yDelta = data(1);
    imageDatasets(i).zDelta = data(3);
    imageDatasets(i).xMinPos = data(2)*imageDatasets(i).xVoxelSize + imageDatasets(i).xMinPos;
    imageDatasets(i).xMaxPos = imageDatasets(i).xMinPos+ imageDatasets(i).xDim*imageDatasets(i).xVoxelSize;
    imageDatasets(i).yMinPos = data(1)*imageDatasets(i).yVoxelSize + imageDatasets(i).yMinPos;
    imageDatasets(i).yMaxPos = imageDatasets(i).yMinPos+ imageDatasets(i).yDim*imageDatasets(i).yVoxelSize;
    imageDatasets(i).zMinPos =  data(3)*imageDatasets(i).zVoxelSize;
    imageDatasets(i).zMaxPos = imageDatasets(i).zMinPos+ imageDatasets(i).zDim*imageDatasets(i).zVoxelSize;
    imageDatasets(i).ParentDelta = find(cellfun(@(x)(~isempty(x)),strfind(names,deltaParent)));
    if isempty(imageDatasets(i).ParentDelta)
        imageDatasets(i).ParentDelta = i;
    end
end
end