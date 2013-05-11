function combineImages()
global imageDatasets rootDir MARGIN newImage outImage datasetName

% root = 'B:\Users\eric_000\Documents\Programming\Chris\';
datasetName = 'CP2';
readMetaData();

readDeltaData(rootDir);


MARGIN = 5;

% %% make mosiac
% %create a dirctory for the new images
if ~isdir(fullfile(rootDir,'Mosiac'))
    mkdir(rootDir,'Mosiac');
end

minXPos = min([imageDatasets(:).xMinPos]);
minYPos = min([imageDatasets(:).yMinPos]);
minZPos = min([imageDatasets(:).zMinPos]);
maxXPos = max([imageDatasets(:).xMaxPos]);
maxYPos = max([imageDatasets(:).yMaxPos]);
maxZPos = max([imageDatasets(:).zMaxPos]);
minXvoxelSize = min([imageDatasets(:).xVoxelSize]);
minYvoxelSize = min([imageDatasets(:).yVoxelSize]);
minZvoxelSize = min([imageDatasets(:).zVoxelSize]);
imageWidth = round((maxXPos-minXPos)/minXvoxelSize +1);
imageHeight = round((maxYPos-minYPos)/minYvoxelSize +1);
imageDepth = round((maxZPos-minZPos)/minZvoxelSize +1);

newImage = cell(length(imageDatasets),1);
outImage = uint8(zeros(imageWidth,imageHeight,imageDepth,min([imageDatasets(:).NumberOfChannels])));
for c=1:min([imageDatasets(:).NumberOfChannels])
    
    fprintf('Chan:%d',c);
    for t=1:min([imageDatasets(:).NumberOfFrames])
        for im=1:length(imageDatasets)
            fprintf('.');
            newImage{im} = uint8(zeros(imageDatasets(im).xDim,imageDatasets(im).yDim,imageDatasets(im).zDim));
            for z=1:imageDatasets(im).zDim
                newImage{im}(:,:,z) = imread(fullfile(rootDir,imageDatasets(im).DatasetName,sprintf('%s_c%d_t%04d_z%04d.tif',imageDatasets(im).DatasetName,c,t,z)));
            end
        end
    end
    fprintf('\n');
    % end
    
    A = zeros(length(imageDatasets),length(imageDatasets));
    
    for i=2:length(imageDatasets)
        A(imageDatasets(i).ParentDelta,i) = 1;
    end
    
%     imageMatch(A,1);
    
    fprintf('Chan:%d',c);
    for t=1:min([imageDatasets(:).NumberOfFrames])
        for im=1:length(imageDatasets)
            fprintf('.');
            for z=1:imageDatasets(im).zDim
                startXind = round((imageDatasets(im).xMinPos-minXPos) / imageDatasets(im).xVoxelSize +1);
                startYind = round((imageDatasets(im).yMinPos-minYPos) / imageDatasets(im).yVoxelSize +1);
                startZind = round((imageDatasets(im).zMinPos-minZPos) / imageDatasets(im).zVoxelSize +1);
                outImage(startXind:startXind+imageDatasets(im).xDim-1,startYind:startYind+imageDatasets(im).yDim-1,startZind+z-1,c)...
                    = newImage{im}(:,:,z);
            end
        end
%         for z=1:size(outImage,3)
%             imwrite(outImage(:,:,z,c),fullfile(rootDir, 'Mosiac', [datasetName sprintf('_c%d_t%04d_z%04d.tif',c,t,z)]),'tif');
%         end
    end
    fprintf('\n');
end

end