function combineImages()
global imageDatasets rootDir MARGIN

readMetaData('B:\Users\eric_000\Documents\Programming\Chris\3mo wmSVZ-2');

readDeltaData('B:\Users\eric_000\Documents\Programming\Chris\3mo wmSVZ-2');
datasetName = '3mo wmSVZ-2';

MARGIN = 5;

% for staticDataInd=1:length(imageDatasets)
%     for otherDataInd=staticDataInd+1:length(imageDatasets)
%         % staticDataInd=1;
%         % otherDataInd=2;
%         %see if these two dataset are suppose to overlap we need to widen
%         deltaX = imageDatasets(otherDataInd).xMinPos - imageDatasets(staticDataInd).xMinPos;
%         deltaY = imageDatasets(otherDataInd).yMinPos - imageDatasets(staticDataInd).yMinPos;
%         
%         if abs(deltaX)>imageDatasets(staticDataInd).xDim*imageDatasets(staticDataInd).xVoxelSize...
%                 || abs(deltaY)>imageDatasets(staticDataInd).yDim*imageDatasets(staticDataInd).yVoxelSize
%             %these images do not overlap
%             %             continue
%             return
%         end
%         
%         tic
%         
%         fprintf(1,'%s <- %s... ',imageDatasets(staticDataInd).DatasetName,imageDatasets(otherDataInd).DatasetName);
%         
%         deltaXpix = round(deltaX / imageDatasets(staticDataInd).xVoxelSize);
%         deltaYpix = round(deltaY / imageDatasets(staticDataInd).yVoxelSize);
%         
%         if deltaXpix>0
%             %Other image is to the right of the static image
%             deltaXpix = max(deltaXpix - MARGIN,1);
%             staticXindStart = deltaXpix;
%             staticXindEnd   = min(imageDatasets(staticDataInd).xDim,imageDatasets(otherDataInd).xDim);
%             otherXindStart  = 1;
%             otherXindEnd    = staticXindEnd-deltaXpix +1;%ensures that the subimages are equal size
%         elseif deltaXpix<0
%             %Other image is to the left of the static image
%             deltaXpix = min(deltaXpix + MARGIN,-1) * -1;
%             staticXindStart = 1;
%             staticXindEnd   = min(imageDatasets(staticDataInd).xDim,imageDatasets(otherDataInd).xDim-deltaXpix);
%             otherXindStart  = imageDatasets(otherDataInd).xDim - staticXindEnd +1;
%             otherXindEnd    = imageDatasets(otherDataInd).xDim;
%         else
%             staticXindStart = 1;
%             staticXindEnd   = min(imageDatasets(staticDataInd).xDim,imageDatasets(otherDataInd).xDim);
%             otherXindStart  = 1;
%             otherXindEnd    = min(imageDatasets(staticDataInd).xDim,imageDatasets(otherDataInd).xDim);
%         end
%         
%         if deltaYpix>0
%             %Other image is to the bottom of the static image
%             deltaYpix = max(deltaYpix - MARGIN,1);
%             staticYindStart = deltaYpix;
%             staticYindEnd   = min(imageDatasets(staticDataInd).yDim,imageDatasets(otherDataInd).yDim);
%             otherYindStart  = 1;
%             otherYindEnd    = staticYindEnd-deltaYpix +1;%ensures that the subimages are equal size
%         elseif deltaYpix<0
%             %Other image is to the top of the static image
%             deltaYpix = min(deltaYpix + MARGIN,-1) * -1;
%             staticYindStart = 1;
%             staticYindEnd   = min(imageDatasets(staticDataInd).yDim,imageDatasets(otherDataInd).yDim-deltaYpix);
%             otherYindStart  = imageDatasets(otherDataInd).yDim - staticYindEnd +1;
%             otherYindEnd    = imageDatasets(otherDataInd).yDim;
%         else
%             staticYindStart = 1;
%             staticYindEnd   = min(imageDatasets(staticDataInd).yDim,imageDatasets(otherDataInd).yDim);
%             otherYindStart  = 1;
%             otherYindEnd    = min(imageDatasets(staticDataInd).yDim,imageDatasets(otherDataInd).yDim);
%         end
%         
%         if 0~=(otherXindEnd-otherXindStart)-(staticXindEnd-staticXindStart)
%             error('X limit mismatch');
%         end
%         if 0~=(otherYindEnd-otherYindStart)-(staticYindEnd-staticYindStart)
%             error('Y limit mismatch');
%         end
%         
%         xOffset = zeros(imageDatasets(staticDataInd).NumberOfChannels);
%         yOffset = zeros(imageDatasets(staticDataInd).NumberOfChannels);
%         maxCorr = zeros(imageDatasets(staticDataInd).NumberOfChannels);
%         for c=1:imageDatasets(staticDataInd).NumberOfChannels
% %             fprintf(1,'c:%d,',c);
%             yDim = min(imageDatasets(staticDataInd).yDim,imageDatasets(otherDataInd).yDim);
%             xDim = min(imageDatasets(staticDataInd).xDim,imageDatasets(otherDataInd).xDim);
%             zDim = min(imageDatasets(staticDataInd).zDim,imageDatasets(otherDataInd).zDim);
%             staticImage = zeros(yDim,xDim,zDim);
%             otherImage = zeros(yDim,xDim,zDim);
%             for z=1:zDim
%                 staticImage(:,:,z) = imread(fullfile(rootDir,imageDatasets(staticDataInd).DatasetName,sprintf('%s_c%d_t%04d_z%04d.tif',imageDatasets(staticDataInd).DatasetName,c,1,z)));
%                 otherImage(:,:,z) = imread(fullfile(rootDir,imageDatasets(otherDataInd).DatasetName,sprintf('%s_c%d_t%04d_z%04d.tif',imageDatasets(otherDataInd).DatasetName,c,1,z)));
%             end
%             staticOverlapImage = staticImage(staticYindStart:staticYindEnd,staticXindStart:staticXindEnd,:);
%             otherOverlapImage = otherImage(otherYindStart:otherYindEnd,otherXindStart:otherXindEnd,:);
%             [xOffset(c) yOffset(c) maxCorr(c)] = registerTwoImages(staticOverlapImage,otherOverlapImage);
%         end
%         
%         avgXOffset = sum(sum(xOffset))/imageDatasets(staticDataInd).NumberOfChannels;
%         avgYOffset = sum(sum(yOffset))/imageDatasets(staticDataInd).NumberOfChannels;
%         avgMaxCorr = sum(sum(maxCorr))/imageDatasets(staticDataInd).NumberOfChannels;
%         
%         xMinPos = imageDatasets(otherDataInd).xMinPos + avgXOffset*imageDatasets(otherDataInd).xVoxelSize;
%         yMinPos = imageDatasets(otherDataInd).yMinPos + avgYOffset*imageDatasets(otherDataInd).yVoxelSize;
%         xMaxPos = xMinPos + imageDatasets(otherDataInd).xVoxelSize*imageDatasets(otherDataInd).xDim;
%         yMaxPos = yMinPos + imageDatasets(otherDataInd).xVoxelSize*imageDatasets(otherDataInd).xDim;
%         
%         imageDatasets(otherDataInd).xMinPos = xMinPos;
%         imageDatasets(otherDataInd).yMinpos = yMinPos;
%         imageDatasets(otherDataInd).xMaxPos = xMaxPos;
%         imageDatasets(otherDataInd).yMaxpos = yMaxPos;
%         
%         %         deltaX = xMinPos - imageDatasets(otherDataInd).xMinPos;
%         %         deltaY = yMinPos - imageDatasets(otherDataInd).yMinPos;
%         sec = toc;
%         fprintf('Changed %s by (%d,%d), maxCorr:%f sec:%f\n',imageDatasets(otherDataInd).DatasetName,avgXOffset,avgYOffset,avgMaxCorr,sec);
%     end
% end

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

for c=4:min([imageDatasets(:).NumberOfChannels])
    for t=1:min([imageDatasets(:).NumberOfFrames])
        newImage = uint8(zeros(imageWidth,imageHeight,imageDepth));
        for im=1:length(imageDatasets)
            for z=1:imageDatasets(im).zDim
                curImage = imread(fullfile(rootDir,imageDatasets(im).DatasetName,sprintf('%s_c%d_t%04d_z%04d.tif',imageDatasets(im).DatasetName,c,t,z)));
                startXind = round((imageDatasets(im).xMinPos-minXPos) / imageDatasets(im).xVoxelSize +1);
                startYind = round((imageDatasets(im).yMinPos-minYPos) / imageDatasets(im).yVoxelSize +1);
                startZind = round((imageDatasets(im).zMinPos-minZPos) / imageDatasets(im).zVoxelSize +1);
                newImage(startXind:startXind+imageDatasets(im).xDim-1,startYind:startYind+imageDatasets(im).yDim-1,startZind+z-1) = curImage;
            end
        end
        for z=1:size(newImage,3)
            imwrite(newImage(:,:,z),fullfile(rootDir, 'Mosiac', [datasetName sprintf('_c%d_t%04d_z%04d.tif',c,t,z)]),'tif');
        end
    end
end

end