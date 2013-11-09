function combineImages()
global imageDatasets rootDir MARGIN newImage outImage datasetName DeltasPresent outImageColor

datasetName = 'DAPI Olig2-514 GFAP-488 Mash1-647 PSA-NCAM-549 lectin-568 22mo wmSVZ';
readMetaData();

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
minXvoxelSize = min([imageDatasets([imageDatasets.zVoxelSize]>0).xVoxelSize]);
minYvoxelSize = min([imageDatasets([imageDatasets.zVoxelSize]>0).yVoxelSize]);
minZvoxelSize = min([imageDatasets([imageDatasets.zVoxelSize]>0).zVoxelSize]);
imageWidth = round((maxXPos-minXPos)/minXvoxelSize +1);
imageHeight = round((maxYPos-minYPos)/minYvoxelSize +1);
imageDepth = round((maxZPos-minZPos)/minZvoxelSize +1);

imageData.DatasetName = datasetName;
imageData.NumberOfChannels = max([imageDatasets(:).NumberOfChannels]);
imageData.NumberOfFrames = max([imageDatasets(:).NumberOfFrames]);
imageData.xDim = imageWidth;
imageData.yDim = imageHeight;
imageData.zDim = imageDepth;
imageData.xVoxelSize = minXvoxelSize;
imageData.yVoxelSize = minYvoxelSize;
imageData.zVoxelSize = minZvoxelSize;

newImage = cell(length(imageDatasets),1);

outImage = zeros(imageWidth,imageHeight,imageDepth,min([imageDatasets(:).NumberOfChannels]),'uint8');
outImageColor = zeros(imageWidth,imageHeight,imageDepth,min([imageDatasets(:).NumberOfChannels]),'uint8');
for c=1:max([imageDatasets(:).NumberOfChannels])
    outImage = zeros(imageWidth,imageHeight,imageDepth,'uint8');
    fprintf('Read Chan:%d',c);
    for t=1:max([imageDatasets(:).NumberOfFrames])
        for datasetIdx=1:length(imageDatasets)
            if (imageDatasets(datasetIdx).NumberOfChannels>=c)
                newImage{datasetIdx} = zeros(imageDatasets(datasetIdx).xDim,imageDatasets(datasetIdx).yDim,imageDatasets(datasetIdx).zDim,'uint8');
                for z=1:imageDatasets(datasetIdx).zDim
                    newImage{datasetIdx}(:,:,z) = imread(fullfile(rootDir,imageDatasets(datasetIdx).DatasetName,sprintf('%s_c%d_t%04d_z%04d.tif',imageDatasets(datasetIdx).DatasetName,c,t,z)));
                end
                fprintf('.');
            end
        end
    end
    fprintf('\n');
    % end
    
%     A = zeros(length(imageDatasets),length(imageDatasets));
%     
%     for i=2:length(imageDatasets)
%         A(imageDatasets(i).ParentDelta,i) = 1;
%     end
    
%     imageMatch(A,1);
    
    fprintf('Making image:%d',c);
    for t=1:max([imageDatasets(:).NumberOfFrames])
        for datasetIdx=1:length(imageDatasets)
            if (imageDatasets(datasetIdx).NumberOfChannels>=c)
                for z=1:imageDatasets(datasetIdx).zDim
                    startXind = round((imageDatasets(datasetIdx).xMinPos-minXPos) / minXvoxelSize +1);
                    startYind = round((imageDatasets(datasetIdx).yMinPos-minYPos) / minYvoxelSize +1);
                    startZind = round((imageDatasets(datasetIdx).zMinPos-minZPos) / minZvoxelSize +1);
                    outImage(startXind:startXind+imageDatasets(datasetIdx).xDim-1,startYind:startYind+imageDatasets(datasetIdx).yDim-1,startZind+z-1)...
                        = newImage{datasetIdx}(:,:,z);
                    outImageColor(startXind:startXind+imageDatasets(datasetIdx).xDim-1,startYind:startYind+imageDatasets(datasetIdx).yDim-1,startZind+z-1)...
                        = datasetIdx;
                end
            end
            clear newImage{datasetIdx}
        end
        
        fprintf('\nWrite Chan:%d',c);
        imwrite(max(outImage(:,:,:),[],3),fullfile(rootDir, prefix, ['_' datasetName sprintf('_c%d_t%04d.tif',c,t)]),'tif','Compression','lzw');
        fprintf('.');
        createMetadata(fullfile(rootDir, prefix),datasetName,imageData);
        modZ = ceil(size(outImage,3)/length(imageDatasets));
        for z=1:size(outImage,3)
            imwrite(outImage(:,:,z),fullfile(rootDir, prefix, [datasetName sprintf('_c%d_t%04d_z%04d.tif',c,t,z)]),'tif','Compression','lzw');
            if (mod(z,modZ)==0)
                fprintf('.');
            end
        end
        
        for reduce=1:5
            fprintf('\nReduce x%d...',reduce);
            imR = CudaMex('ReduceImage',outImage,[reduce,reduce,1]);
            fprintf(' done. Writing:');
            imDataReduced = imageData;
            imDataReduced.xDim = size(imR,2);
            imDataReduced.yDim = size(imR,1);
            imDataReduced.zDim = size(imR,3);
            imDataReduced.xVoxelSize = imageData.xVoxelSize*reduce;
            imDataReduced.yVoxelSize = imageData.yVoxelSize*reduce;
            % ZPixelPhysicalSize is same as orginal
            
            if ~isdir(fullfile(rootDir,prefix,['x' num2str(reduce)]))
                mkdir(fullfile(rootDir,prefix),['x' num2str(reduce)]);
            end
            
            createMetadata(fullfile(rootDir, prefix, ['x' num2str(reduce)]),datasetName,imDataReduced);
            for z=1:size(outImage,3)
                imwrite(imR(:,:,z),fullfile(rootDir, prefix, ['x' num2str(reduce)], [datasetName sprintf('_c%d_t%04d_z%04d.tif',c,t,z)]),'tif','Compression','lzw');
                if (mod(z,modZ)==0)
                    fprintf('.');
                end
            end
            clear imR;
        end
    end
    fprintf('\n');
    clear outImage;
    clear outImageColor;
end

clear mex
end