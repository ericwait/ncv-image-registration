function [ imageData, varargout ] = GetFinalSize( imageDatasets, datasetName )
%GETFINALSIZE Takes all of the sub-image datasets and returns what the
%final dataset should look like.
%   imageData is the metadata that should be used to create the montage
%   tiff.
%   Optionally minPos and maxPos can be returned, such as
%   [ imageData, minPos, maxPos ] = GetFinalSize( imageDatasets )
%   min and max are [X,Y]

minXPos = min([imageDatasets(:).xMinPos]);
minYPos = min([imageDatasets(:).yMinPos]);
minZPos = min([imageDatasets(:).zMinPos]);
maxXPos = max([imageDatasets(:).xMaxPos]);
maxYPos = max([imageDatasets(:).yMaxPos]);
maxZPos = max([imageDatasets(:).zMaxPos]);
minXvoxelSize = min([imageDatasets([imageDatasets.ZPixelPhysicalSize]>0).XPixelPhysicalSize]);
minYvoxelSize = min([imageDatasets([imageDatasets.ZPixelPhysicalSize]>0).YPixelPhysicalSize]);
minZvoxelSize = min([imageDatasets([imageDatasets.ZPixelPhysicalSize]>0).ZPixelPhysicalSize]);
imageWidth = length(minXPos:maxXPos);
imageHeight = length(minYPos:maxYPos);
imageDepth = length(minZPos:maxZPos);

imageData.DatasetName = datasetName;
imageData.NumberOfChannels = max([imageDatasets(:).NumberOfChannels]);
imageData.NumberOfFrames = max([imageDatasets(:).NumberOfFrames]);
imageData.XDimension = imageWidth;
imageData.YDimension = imageHeight;
imageData.ZDimension = imageDepth;
imageData.XPixelPhysicalSize = minXvoxelSize;
imageData.YPixelPhysicalSize = minYvoxelSize;
imageData.ZPixelPhysicalSize = minZvoxelSize;

if (nargout>1)
    varargout{1} = [minXPos,minYPos,minZPos];
end
if (nargout>2)
    varargout{2} = [imageWidth,imageHeight,imageDepth];
end
end

