global imageDatasets outImage

hold off
img = max(outImage,[],3);
imagesc(img), colormap gray
hold on

cmap = hsv(length(imageDatasets));

for i=1:length(imageDatasets)
%     rectangle('Position',...
%         [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).yVoxelSize...
%         (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).xVoxelSize...
%         imageDatasets(i).yDim imageDatasets(i).xDim],'EdgeColor','r');
%     rectangle('Position',...
%         [(imageDatasets(i).yMinPos-minYPos-imageDatasets(i).yDelta)/imageDatasets(i).yVoxelSize...
%         (imageDatasets(i).xMinPos-minXPos-imageDatasets(i).xDelta)/imageDatasets(i).xVoxelSize...
%         imageDatasets(i).yDim imageDatasets(i).xDim],'EdgeColor','g');
%     
    centerCur = [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).yVoxelSize + imageDatasets(i).yDim/2 ...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).xVoxelSize + imageDatasets(i).xDim/2];
    
    centerParent = [(imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos)/imageDatasets(imageDatasets(i).ParentDelta).yVoxelSize + imageDatasets(imageDatasets(i).ParentDelta).yDim/2 ...
        (imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos)/imageDatasets(imageDatasets(i).ParentDelta).xVoxelSize + imageDatasets(imageDatasets(i).ParentDelta).xDim/2];
    
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b');
end
