function alignmentTests()
global imageDatasets rootDir MARGIN newImage outImage datasetName DeltasPresent
%% start
datasetName = 'DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568';
readMetaData();

if (isempty(imageDatasets))
    error('No images for dataset %s\n',datasetName);
end

readDeltaData(rootDir);

MARGIN = 5;

%% redo
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

newImage = cell(length(imageDatasets),1);

% outImage = zeros(imageWidth,imageHeight,imageDepth,min([imageDatasets(:).NumberOfChannels]),'uint8');
% for c=2:max([imageDatasets(:).NumberOfChannels])
c=2;
    outImage = zeros(imageWidth,imageHeight,imageDepth,'uint8');
    fprintf('Read Chan:%d',c);
    for t=1:min([imageDatasets(:).NumberOfFrames])
        for im=1:length(imageDatasets)
            if (imageDatasets(im).NumberOfChannels>=c)
                newImage{im} = zeros(imageDatasets(im).xDim,imageDatasets(im).yDim,imageDatasets(im).zDim,'uint8');
                for z=1:imageDatasets(im).zDim
                    newImage{im}(:,:,z) = imread(fullfile(rootDir,imageDatasets(im).DatasetName,...
                        sprintf('%s_c%d_t%04d_z%04d.tif',imageDatasets(im).DatasetName,c,t,z)));
                end
                fprintf('.');
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
    
    fprintf('Making image:%d',c);
    for t=1:min([imageDatasets(:).NumberOfFrames])
        for im=1:length(imageDatasets)
            if (imageDatasets(im).NumberOfChannels>=c)
                for z=1:imageDatasets(im).zDim
                    startXind = round((imageDatasets(im).xMinPos-minXPos) / minXvoxelSize +1);
                    startYind = round((imageDatasets(im).yMinPos-minYPos) / minYvoxelSize +1);
                    startZind = round((imageDatasets(im).zMinPos-minZPos) / minZvoxelSize +1);
                    outImage(startXind:startXind+imageDatasets(im).xDim-1,startYind:startYind+imageDatasets(im).yDim-1,...
                        startZind+z-1) = newImage{im}(:,:,z);
                end
            end
        end        
    end
%end
testingDeltas();
end