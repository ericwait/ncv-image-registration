function imageMatch(A,curNode)
global imageDatasets newImage
children = find(A(curNode,:));
for i=1:length(children)
    parentXstart = round(max(0,imageDatasets(children(i)).xMinPos - imageDatasets(curNode).xMinPos)...
        /imageDatasets(curNode).xVoxelSize +1);
    
    parentXend = round(min(imageDatasets(curNode).xDim,(imageDatasets(children(i)).xMaxPos-imageDatasets(curNode).xMinPos)...
        /imageDatasets(curNode).xVoxelSize));
    
    parentYstart = round(max(0,imageDatasets(children(i)).yMinPos - imageDatasets(curNode).yMinPos)...
        /imageDatasets(curNode).yVoxelSize +1);
    
    parentYend = round(min(imageDatasets(curNode).yDim, (imageDatasets(children(i)).yMaxPos-imageDatasets(curNode).yMinPos)...
        /imageDatasets(curNode).yVoxelSize));
    
    parentZstart = round(max(0,imageDatasets(children(i)).zMinPos - imageDatasets(curNode).zMinPos)...
        /imageDatasets(curNode).zVoxelSize +1);
    
    parentZend = round(min(imageDatasets(curNode).zDim, (imageDatasets(children(i)).zMaxPos-imageDatasets(curNode).zMinPos)...
        /imageDatasets(curNode).zVoxelSize));
    
    childXstart = round(max(0,...
        imageDatasets(curNode).xMinPos - imageDatasets(children(i)).xMinPos)...
        /imageDatasets(curNode).xVoxelSize +1);
    
    childXend = round(min(imageDatasets(children(i)).xDim, (imageDatasets(curNode).xMaxPos-imageDatasets(children(i)).xMinPos)...
        /imageDatasets(curNode).xVoxelSize));
    
    childYstart = round(max(0,imageDatasets(curNode).yMinPos - imageDatasets(children(i)).yMinPos)...
        /imageDatasets(curNode).yVoxelSize +1);
    
    childYend = round(min(imageDatasets(children(i)).yDim, (imageDatasets(curNode).yMaxPos-imageDatasets(children(i)).yMinPos)...
        /imageDatasets(curNode).yVoxelSize));
    
    childZstart = round(max(0,imageDatasets(curNode).zMinPos - imageDatasets(children(i)).zMinPos)...
        /imageDatasets(curNode).zVoxelSize +1);
    
    childZend = round(min(imageDatasets(children(i)).zDim, (imageDatasets(curNode).zMaxPos-imageDatasets(children(i)).zMinPos)...
        /imageDatasets(curNode).zVoxelSize));
    
    parentIm = newImage{curNode}(parentXstart:parentXend,parentYstart:parentYend,parentZstart:parentZend);
    childIm = newImage{children(i)}(childXstart:childXend,childYstart:childYend,childZstart:childZend);
    
    X = [double(childIm(:)) ones(length(childIm(:)),1)];
    b = X\double(parentIm(:));
    
%     newImage{children(i)} = newImage{children(i)}*b(1) + b(2);
newImage{children(i)} = newImage{children(i)}*1 + 0;
    
    imageMatch(A,children(i));
end
end