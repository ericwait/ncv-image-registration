function combineImages()
global imageDatasets DeltasPresent
totalTime = tic;
imageDatasets = [];

datasetName = 'DAPI Olig2-514 GFAP-488 Mash1-647 PSA-NCAM-549 lectin-568 22mo wmSVZ';

rootDir = uigetdir('');
if rootDir==0, return, end
dlist = dir(rootDir);

for i=1:length(dlist)
    if (~isdir(fullfile(rootDir,dlist(i).name)) || strcmp('..',dlist(i).name) || strcmp('.',dlist(i).name) ||...
            ~isempty(strfind(dlist(i).name,'Montage')))
        continue;
    end
    if isempty(imageDatasets)
        imageDatasets = readMetaData(fullfile(rootDir,dlist(i).name));
    else
        imageDatasets(length(imageDatasets)+1) = readMetaData(fullfile(rootDir,dlist(i).name));
    end
end

if (isempty(imageDatasets))
    error('No images for dataset %s\n',datasetName);
end

readDeltaData(rootDir);

%% save out
MARGIN = 5;

if DeltasPresent==1
    prefix = [datasetName '_Montage_wDelta'];
else
    prefix = [datasetName '_Montage'];
end

% %% make mosiac
% %create a dirctory for the new images
if ~isdir(fullfile(rootDir,prefix))
    mkdir(rootDir,prefix);
end

minXPos = min([imageDatasets(:).xMinPos]);
minYPos = min([imageDatasets(:).yMinPos]);
minZPos = min([imageDatasets(:).zMinPos]);
maxXPos = max([imageDatasets(:).xMaxPos]);
maxYPos = max([imageDatasets(:).yMaxPos]);
maxZPos = max([imageDatasets(:).zMaxPos]);
minXvoxelSize = min([imageDatasets([imageDatasets.ZPixelPhysicalSize]>0).XPixelPhysicalSize]);
minYvoxelSize = min([imageDatasets([imageDatasets.ZPixelPhysicalSize]>0).YPixelPhysicalSize]);
minZvoxelSize = min([imageDatasets([imageDatasets.ZPixelPhysicalSize]>0).ZPixelPhysicalSize]);
imageWidth = round((maxXPos-minXPos)/minXvoxelSize +1);
imageHeight = round((maxYPos-minYPos)/minYvoxelSize +1);
imageDepth = round((maxZPos-minZPos)/minZvoxelSize +1);

imageData.DatasetName = datasetName;
imageData.NumberOfChannels = max([imageDatasets(:).NumberOfChannels]);
imageData.NumberOfFrames = max([imageDatasets(:).NumberOfFrames]);
imageData.XDimension = imageWidth;
imageData.YDimension = imageHeight;
imageData.ZDimension = imageDepth;
imageData.XPixelPhysicalSize = minXvoxelSize;
imageData.YPixelPhysicalSize = minYvoxelSize;
imageData.ZPixelPhysicalSize = minZvoxelSize;

% for chan=1:imageData.NumberOfChannels
chan=4
    chanStart = tic;
    outImage = zeros(imageWidth,imageHeight,imageDepth,'uint8');
    outImageColor = zeros(imageWidth,imageHeight,imageDepth,'uint8');
    fprintf('Chan:%d\n',chan);
    for datasetIdx=1:length(imageDatasets)
        if (imageDatasets(datasetIdx).NumberOfChannels>=chan)
            startXind = round((imageDatasets(datasetIdx).xMinPos-minXPos) / minXvoxelSize +1);
            startYind = round((imageDatasets(datasetIdx).yMinPos-minYPos) / minYvoxelSize +1);
            startZind = round((imageDatasets(datasetIdx).zMinPos-minZPos) / minZvoxelSize +1);
            
            outImage(startXind:startXind+imageDatasets(datasetIdx).XDimension-1,...
                startYind:startYind+imageDatasets(datasetIdx).YDimension-1,...
                startZind:startZind+imageDatasets(datasetIdx).ZDimension-1)...
                = tiffReader('uint8',chan,1,[],fullfile(rootDir,imageDatasets(datasetIdx).DatasetName));
            
            outImageColor(startXind:startXind+imageDatasets(datasetIdx).XDimension-1,...
                startYind:startYind+imageDatasets(datasetIdx).YDimension-1,...
                startZind:startZind+imageDatasets(datasetIdx).ZDimension-1) = ones(imageDatasets(datasetIdx).XDimension,...
                imageDatasets(datasetIdx).YDimension,imageDatasets(datasetIdx).ZDimension)*datasetIdx;
        end
    end
    
    imwrite(max(outImage(:,:,:),[],3),fullfile(rootDir, prefix, ['_' datasetName sprintf('_c%d_t%04d.tif',chan,1)]),'tif','Compression','lzw');
    createMetadata(fullfile(rootDir, prefix),imageData);
    modZ = ceil(size(outImage,3)/length(imageDatasets));
    for z=1:size(outImage,3)
        imwrite(outImage(:,:,z),fullfile(rootDir, prefix, [datasetName sprintf('_c%d_t%04d_z%04d.tif',chan,1,z)]),'tif','Compression','lzw');
        if (mod(z,modZ)==0)
            fprintf('.');
        end
    end
    
    %testingDeltas();
    
    maxReduction = ceil(max(size(outImage))/1024);
    
    for reduce=1:maxReduction
        fprintf('\nReduce x%d...',reduce);
        imR = CudaMex('ReduceImage',outImage,[reduce,reduce,1]);
        imDataReduced = imageData;
        imDataReduced.XDimension = size(imR,1);
        imDataReduced.YDimension = size(imR,2);
        imDataReduced.ZDimension = size(imR,3);
        imDataReduced.XPixelPhysicalSize = imageData.XPixelPhysicalSize*reduce;
        imDataReduced.YPixelPhysicalSize = imageData.YPixelPhysicalSize*reduce;
        % ZPixelPhysicalSize is same as orginal
        
        if ~isdir(fullfile(rootDir,prefix,['x' num2str(reduce)]))
            mkdir(fullfile(rootDir,prefix),['x' num2str(reduce)]);
        end
        
        createMetadata(fullfile(rootDir, prefix, ['x' num2str(reduce)]),imDataReduced);
        for z=1:size(outImage,3)
            imwrite(imR(:,:,z),fullfile(rootDir, prefix, ['x' num2str(reduce)], [datasetName sprintf('_c%d_t%04d_z%04d.tif',chan,1,z)]),'tif','Compression','lzw');
            if (mod(z,modZ)==0)
                fprintf('.');
            end
        end
        
        fprintf(' done.\n');
        clear imR;
    end
    
    clear outImage;
    clear outImageColor;
    fprintf('Chan:%d done in %f sec\n',chan,toc(chanStart));
%end
fprintf('Completed in %f sec\n',toc(totalTime));

clear mex