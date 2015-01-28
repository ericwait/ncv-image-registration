function combineImages()
totalTime = tic;
imageDatasets = [];
deltasPresent = 0;

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
        [imageDatasets,~] = readMetaData(fullfile(pathName,dirNames{1}{i},[dirNames{1}{i},'.txt']));
    else
        [imageDatasets(end+1),~] = readMetaData(fullfile(pathName,dirNames{1}{i},[dirNames{1}{i},'.txt']));
    end
end

if (isempty(imageDatasets))
    error('No images for dataset %s\n',datasetName);
end

logDir = fullfile(imageDatasets(1).imageDir,'..','_GraphLog');
if (~exist(logDir,'dir'))
    mkdir(logDir);
end

[deltasPresent,imageDatasets] = readDeltaData(pathName,imageDatasets);
if (0~=deltasPresent)
    refine = questdlg('Would you like to use the old registration delta?','Refine Deltas?','Old','Redo','Old');
    if (refine==0)
        return
    elseif (strcmp(refine,'Redo'))
        deltasPresent = 0;
        
        for i=1:length(imageDatasets)
            imageDatasets(i).ParentDelta = 0;
            imageDatasets(i).Children = [];
            imageDatasets(i).xMinPos = 0;
            imageDatasets(i).yMinPos = 0;
            imageDatasets(i).zMinPos = 0;
            imageDatasets(i).xMaxPos = 0;
            imageDatasets(i).yMaxPos = 0;
            imageDatasets(i).zMaxPos = 0;
        end
    end
end

if (0==deltasPresent)
    refine = questdlg('Would you like to refine registration or use microscope data?','Refine Deltas?','Refine','Microscope','Refine W/ Visualizer','Microscope');
    if (refine==0), return, end
else
    prefix = [datasetName '_Montage_wDelta'];
    refine = '';
end

visualize = questdlg('Would you like to see the results?','Results Visualizer','Yes','No','Visualize Only','No');
if (isempty(visualize)), return, end
numCudaDevices = CudaMex('DeviceCount');
if (strcmp(visualize,'Visualize Only')==0 && numCudaDevices>0)
    reducIms = questdlg('Would you like to create reduced images?','Reduce Images','Yes','No','No');
    if (isempty(reducIms)), return, end
else
    reducIms = 'No';
end

if (strcmp(refine,'Refine') || strcmp(refine,'Refine W/ Visualizer'))
    prefix = [datasetName '_Montage_wDelta'];
    imageDatasets = createDeltas(imageDatasets,strcmp(refine,'Refine W/ Visualizer'));
    [~,imageDatasets] = readDeltaData(pathName,imageDatasets);
elseif (0==deltasPresent)
    prefix = [datasetName '_Montage'];
end

%% make mosiac
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

tmpImageData = imageData;

[im,~] = tiffReader(fullfile(pathName,dirNames{1}{i},[dirNames{1}{i} '.txt']));
w = whos('im');
clear im

for chan=1:imageData.NumberOfChannels
    chanStart = tic;
    outImage = zeros(imageHeight,imageWidth,imageDepth,w.class);
    if (strcmp(visualize,'No')==0)
        outImageColor = zeros(imageHeight,imageWidth,imageDepth,w.class);
    end
    fprintf('Chan:%d\n',chan);
    for datasetIdx=1:length(imageDatasets)
        if (imageDatasets(datasetIdx).NumberOfChannels>=chan)
            startXind = round((imageDatasets(datasetIdx).xMinPos-minXPos) / minXvoxelSize +1);
            startYind = round((imageDatasets(datasetIdx).yMinPos-minYPos) / minYvoxelSize +1);
            startZind = round((imageDatasets(datasetIdx).zMinPos-minZPos) / minZvoxelSize +1);
            
            [nextIm,~] = tiffReader(fullfile(pathName,dirNames{1}{datasetIdx},[dirNames{1}{datasetIdx} '.txt']),[],chan,[],[],[],true);
            fprintf('.');
            
            roi = [startXind,startYind,startZind,...
                startXind+min(imageDatasets(datasetIdx).XDimension,size(nextIm,1))-1,...
                startYind+min(imageDatasets(datasetIdx).YDimension,size(nextIm,2))-1,...
                startZind+min(imageDatasets(datasetIdx).ZDimension,size(nextIm,3))-1];
            
            outRoi = outImage(roi(2):roi(5),roi(1):roi(4),roi(3):roi(6));
            difInd = outRoi>0;
            nextSum = sum(sum(sum(nextIm(difInd))));
            outSum = sum(sum(sum(outRoi(difInd))));
            
            if outSum>nextSum
                nextIm(difInd) = outRoi(difInd);
            end
            
            clear outRoi
            
            outImage(roi(2):roi(5),roi(1):roi(4),roi(3):roi(6)) = nextIm;
            
            clear nextIm
            
            if (strcmp(visualize,'No')==0)
                outImageColor(startYind:startYind+imageDatasets(datasetIdx).YDimension-1,...
                    startXind:startXind+imageDatasets(datasetIdx).XDimension-1,...
                    startZind:startZind+imageDatasets(datasetIdx).ZDimension-1) = ones(imageDatasets(datasetIdx).YDimension,...
                    imageDatasets(datasetIdx).XDimension,imageDatasets(datasetIdx).ZDimension)*datasetIdx;
            end
        end
    end
    
    if (strcmp(visualize,'Yes') || strcmp(visualize,'Visualize Only'))
        figure,imagesc(max(outImage,[],3)),colormap gray, axis image
        title(sprintf('Cannel:%d',chan),'Interpreter','none','Color','w');
        testingDeltas(outImage, outImageColor,imageDatasets,chan);
    else
        [fig,ax] = testingDeltas(outImage,[],imageDatasets,chan);
        set(fig,'Units','normalized','Position',[0 0 1 1]);
        frm = getframe(ax);
        imwrite(frm.cdata,fullfile(logDir,sprintf('%s_c%02d_minSpanTree.tif',datasetName,chan)),'tif','Compression','lzw');
        close(fig);
    end
    
    w = whos('outImage');
    userview = memory;
    
    if (size(outImage,1)>size(outImage,2) && userview.MemAvailableAllArrays>w.bytes)
        outImage = permute(outImage(end:-1:1,:,:),[2,1,3]);
        tmpImageData.XDimension = imageData.YDimension;
        tmpImageData.YDimension = imageData.XDimension;
    end
    
    if (strcmp(visualize,'Visualize Only')==0)
        imwrite(max(outImage,[],3),fullfile(pathName, prefix, ['_' datasetName sprintf('_c%02d_t%04d.tif',chan,1)]),'tif','Compression','lzw');
        createMetadata(fullfile(pathName, prefix),tmpImageData);
        modZ = ceil(size(outImage,3)/length(imageDatasets));
        for z=1:size(outImage,3)
            imwrite(outImage(:,:,z),fullfile(pathName, prefix, [datasetName sprintf('_c%02d_t%04d_z%04d.tif',chan,1,z)]),'tif','Compression','lzw');
            if (mod(z,modZ)==0)
                fprintf('.');
            end
        end
    end
    
    if (strcmp(reducIms,'Yes'))
        maxReduction = ceil(size(outImage,2)/2048);
        
        w = whos('outImage');
        
        if (userview.MemAvailableAllArrays>w.bytes*max((numCudaDevices-1),1))% -1 because one of the pools will use the current copy
            numPools = max(numCudaDevices,1);
        else
            numPools = 1;
        end
        
        poolObj = gcp('nocreate');
        if (isempty(poolObj))
            poolObj = parpool(numPools);
        elseif (poolObj.NumWorkers>numPools)
            delete(poolObj);
            poolObj = parpool(numPools);
        end

        if (isempty(poolObj)), error('No pool for Cuda Reduce!'); end
        
        spmd
            for reduce=labindex:numlabs:maxReduction
                if (reduce==1)
                    fprintf('\nReduce x%d...',reduce);
                    imR = outImage;
                    imDataReduced = tmpImageData;
                else
%                     device = mod(reduce,numCudaDevices)+1;
                    device = 1;
                    fprintf('\nReduce x%d...',reduce);
                    imR = CudaMex('ReduceImage',outImage,[reduce,reduce,1],'mean',device);
                    imDataReduced = tmpImageData;
                    imDataReduced.XDimension = size(imR,2);
                    imDataReduced.YDimension = size(imR,1);
                    imDataReduced.ZDimension = size(imR,3);
                    imDataReduced.XPixelPhysicalSize = tmpImageData.XPixelPhysicalSize*reduce;
                    imDataReduced.YPixelPhysicalSize = tmpImageData.YPixelPhysicalSize*reduce;
                    % ZPixelPhysicalSize is same as orginal
                end
                
                if ~isdir(fullfile(pathName,prefix,['x' num2str(reduce)]))
                    mkdir(fullfile(pathName,prefix),['x' num2str(reduce)]);
                end
                
                createMetadata(fullfile(pathName, prefix, ['x' num2str(reduce)]),imDataReduced);
                for z=1:size(outImage,3)
                    imwrite(imR(:,:,z),fullfile(pathName, prefix, ['x' num2str(reduce)], [datasetName sprintf('_c%02d_t%04d_z%04d.tif',chan,1,z)]),'tif','Compression','lzw');
                    if (mod(z,modZ)==0)
                        fprintf('.');
                    end
                end
                
                fprintf(' done.\n');
            end
        end
    end
    
    clear outImage;
    if (strcmp(visualize,'No')~=0)
        clear outImageColor;
    end
    tm = toc(chanStart);
    fprintf('Chan:%d done in %s\n',chan,printTime(tm));
end

colorMip = colorMIP(fullfile(pathName, prefix,[tmpImageData.DatasetName '.txt']));
f = figure;
imagesc(colorMip);%,'Parent',ax);
ax = get(f,'CurrentAxes');
drawBoxesLines(f,ax,imageDatasets,0,tmpImageData);
frm = getframe(ax);
imwrite(frm.cdata,fullfile(pathName,prefix,sprintf('_%s_graph.tif',tmpImageData.DatasetName)),'tif','Compression','lzw');
close(f);
tm = toc(totalTime);
poolObj = gcp('nocreate');
if (~isempty(poolObj))
    delete(poolObj);
end
fprintf('Completed in %s\n',printTime(tm));

clear mex
