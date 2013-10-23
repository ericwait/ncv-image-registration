global imageDatasets outImage
figure
hold off
img = max(outImage(:,:,:),[],3);
imagesc(img), colormap gray
hold on

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

cmap = hsv(length(imageDatasets));

for i=1:length(imageDatasets)
%         rectangle('Position',...
%         [(imageDatasets(i).yMinPos-minYPos-imageDatasets(i).yDelta)/imageDatasets(i).yVoxelSize...
%         (imageDatasets(i).xMinPos-minXPos-imageDatasets(i).xDelta)/imageDatasets(i).xVoxelSize...
%         imageDatasets(i).yDim imageDatasets(i).xDim],'EdgeColor','b','LineStyle','-.');
    rectangle('Position',...
        [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).yVoxelSize...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).xVoxelSize...
        imageDatasets(i).yDim imageDatasets(i).xDim],'EdgeColor','r','LineStyle',':');
    
    centerCur = [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).yVoxelSize + imageDatasets(i).yDim/2 ...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).xVoxelSize + imageDatasets(i).xDim/2];
    
    text(centerCur(1)+20,centerCur(2)-20,num2str(i),'color','g');
    
    centerParent = [(imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos)/imageDatasets(imageDatasets(i).ParentDelta).yVoxelSize + imageDatasets(imageDatasets(i).ParentDelta).yDim/2 ...
        (imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos)/imageDatasets(imageDatasets(i).ParentDelta).xVoxelSize + imageDatasets(imageDatasets(i).ParentDelta).xDim/2];
    
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'g');
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'g');
end

