function combineImages()
totalTime = tic;
imageDatasets = [];
DeltasPresent = 0;

[fileName,pathName,~] = uigetfile('.txt');

if fileName==0, return, end

fHand = fopen(fullfile(pathName,fileName),'rt');
dirNames = textscan(fHand,'%s','delimiter','\n');
fclose(fHand);

datasetName = [];
str1 = dirNames{1}{1};
for i=length(str1):-1:1
    bmatch = strncmpi(str1(1:i),dirNames{1},i);
    if nnz(bmatch)==length(dirNames{1})
        datasetName = str1(1:i);
        break
    end
end

for i=1:length(dirNames{1})
    if isempty(imageDatasets)
        [imageDatasets,~] = readMetaData(fullfile(pathName,dirNames{1}{i}));
    else
        [imageDatasets(end+1),~] = readMetaData(fullfile(pathName,dirNames{1}{i}));
    end
end

if (isempty(imageDatasets))
    error('No images for dataset %s\n',datasetName);
end

%deltasPresent = 0;
[deltasPresent,imageDatasets] = readDeltaData(pathName,imageDatasets);

if (0==deltasPresent)
    result = questdlg('Would you like to refine registration or use microscope data?','Refine Deltas?','Refine','Microscope','Microscope');
    switch result
        case 'Refine'
            prefix = [datasetName '_Montage_wDelta'];
            answer = inputdlg('Channel to register:','Channel Chooser',1,{'3'});
            imageDatasets = createDeltas(imageDatasets,str2double(answer));
        otherwise
            prefix = [datasetName '_Montage'];
    end
else
    prefix = [datasetName '_Montage_wDelta'];
end

answer = questdlg('Would you like to see the results?','Results Visualizer','Yes','No','No');

%% save out
MARGIN = 5;


% %% make mosiac
% %create a dirctory for the new images
if ~isdir(fullfile(pathName,prefix))
    mkdir(pathName,prefix);
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

[im,~] = tiffReader([],[],[],[],fullfile(pathName,dirNames{1}{i}));
w = whos('im');
clear im

%for chan=1:imageData.NumberOfChannels
for chan=[3,5]
    chanStart = tic;
    outImage = zeros(imageWidth,imageHeight,imageDepth,w.class);
    outImageColor = zeros(imageWidth,imageHeight,imageDepth,w.class);
    fprintf('Chan:%d\n',chan);
    for datasetIdx=1:length(imageDatasets)
        if (imageDatasets(datasetIdx).NumberOfChannels>=chan)
            startXind = round((imageDatasets(datasetIdx).xMinPos-minXPos) / minXvoxelSize +1);
            startYind = round((imageDatasets(datasetIdx).yMinPos-minYPos) / minYvoxelSize +1);
            startZind = round((imageDatasets(datasetIdx).zMinPos-minZPos) / minZvoxelSize +1);
            
            [nextIm,~] = tiffReader([],chan,[],[],fullfile(pathName,dirNames{1}{datasetIdx}));
            
            roi = [startXind,startYind,startZind,...
                startXind+min(imageDatasets(datasetIdx).XDimension,size(nextIm,1))-1,...
                startYind+min(imageDatasets(datasetIdx).YDimension,size(nextIm,2))-1,...
                startZind+min(imageDatasets(datasetIdx).ZDimension,size(nextIm,3))-1];
            
            outRoi = outImage(roi(1):roi(4),roi(2):roi(5),roi(3):roi(6));
            difInd = outRoi>0;
            nextSum = sum(sum(sum(nextIm(difInd))));
            outSum = sum(sum(sum(outRoi(difInd))));
            
            if outSum>nextSum
                nextIm(difInd) = outRoi(difInd);
            end
            
            clear outRoi
            
            outImage(roi(1):roi(4),roi(2):roi(5),roi(3):roi(6)) = nextIm;
            
            clear nextIm
            
            outImageColor(startXind:startXind+imageDatasets(datasetIdx).XDimension-1,...
                startYind:startYind+imageDatasets(datasetIdx).YDimension-1,...
                startZind:startZind+imageDatasets(datasetIdx).ZDimension-1) = ones(imageDatasets(datasetIdx).XDimension,...
                imageDatasets(datasetIdx).YDimension,imageDatasets(datasetIdx).ZDimension)*datasetIdx;
        end
    end
    
%     imwrite(max(outImage(:,:,:),[],3),fullfile(pathName, prefix, ['_' datasetName sprintf('_c%02d_t%04d.tif',chan,1)]),'tif','Compression','lzw');
%     createMetadata(fullfile(pathName, prefix),imageData);
%     modZ = ceil(size(outImage,3)/length(imageDatasets));
%     for z=1:size(outImage,3)
%         imwrite(outImage(:,:,z),fullfile(pathName, prefix, [datasetName sprintf('_c%02d_t%04d_z%04d.tif',chan,1,z)]),'tif','Compression','lzw');
%         if (mod(z,modZ)==0)
%             fprintf('.');
%         end
%     end
    
    if (strcmp(answer,'Yes'))
        figure,imagesc(max(outImage,[],3)),colormap gray, axis image
        testingDeltas(outImage, outImageColor,imageDatasets);
    end
    
%     maxReduction = ceil(max(size(outImage))/1024);
%     
%     for reduce=1:maxReduction
%         fprintf('\nReduce x%d...',reduce);
%         imR = CudaMex('ReduceImage',outImage,[reduce,reduce,1]);
%         imDataReduced = imageData;
%         imDataReduced.XDimension = size(imR,1);
%         imDataReduced.YDimension = size(imR,2);
%         imDataReduced.ZDimension = size(imR,3);
%         imDataReduced.XPixelPhysicalSize = imageData.XPixelPhysicalSize*reduce;
%         imDataReduced.YPixelPhysicalSize = imageData.YPixelPhysicalSize*reduce;
%         % ZPixelPhysicalSize is same as orginal
%         
%         if ~isdir(fullfile(pathName,prefix,['x' num2str(reduce)]))
%             mkdir(fullfile(pathName,prefix),['x' num2str(reduce)]);
%         end
%         
%         createMetadata(fullfile(pathName, prefix, ['x' num2str(reduce)]),imDataReduced);
%         for z=1:size(outImage,3)
%             imwrite(imR(:,:,z),fullfile(pathName, prefix, ['x' num2str(reduce)], [datasetName sprintf('_c%02d_t%04d_z%04d.tif',chan,1,z)]),'tif','Compression','lzw');
%             if (mod(z,modZ)==0)
%                 fprintf('.');
%             end
%         end
%         
%         fprintf(' done.\n');
%         clear imR;
%     end
    
    clear outImage;
    clear outImageColor;
    fprintf('Chan:%d done in %f sec\n',chan,toc(chanStart));
end
fprintf('Completed in %f sec\n',toc(totalTime));

clear mex