global imageDatasets outImage outImageColor
figure
hold off
[img, colorIdx] = max(outImage(:,:,:),[],3);
[r,c] = ndgrid(1:size(outImage,1),1:size(outImage,2));
idx = sub2ind(size(outImage),r,c,colorIdx);
pixelColors = outImageColor(idx);
cmap = hsv(12);
cmap = cmap(randi(12, max(double(pixelColors(:)))+1),:);
alpha = cat(3, reshape(cmap(pixelColors+1,1),size(pixelColors)), reshape(cmap(pixelColors+1,2),size(pixelColors)), reshape(cmap(pixelColors+1,3),size(pixelColors)));
imgC = alpha .* repmat(mat2gray(img),[1 1 3]);

% imagesc(img), colormap gray
image(imgC);
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

