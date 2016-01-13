function [ imageData, minPos, imageSize ] = GetFinalSize( imageDatasets, datasetName )
%GETFINALSIZE Takes all of the sub-image datasets and returns what the
%final dataset should look like.
%   imageData is the metadata that should be used to create the montage
%   tiff.
%   Optionally minPos and maxPos can be returned, such as
%   [ imageData, minPos, maxPos ] = GetFinalSize( imageDatasets )
%   min and max are [X,Y]

imageData = imageDatasets(1);

minPos = min(vertcat(imageDatasets(:).MinPos));
maxPos = max(vertcat(imageDatasets(:).MaxPos));
minVoxelSize = min(vertcat(imageDatasets(:).PixelPhysicalSize));

imageWidth = length(minPos(1):maxPos(1));
imageHeight = length(minPos(2):maxPos(2));
imageDepth = length(minPos(3):maxPos(3));

imageData.DatasetName = datasetName;
imageData.NumberOfChannels = max([imageDatasets(:).NumberOfChannels]);
imageData.NumberOfFrames = max([imageDatasets(:).NumberOfFrames]);
imageData.Dimensions = [imageWidth,imageHeight,imageDepth];
imageData.PixelPhysicalSize = minVoxelSize;

imageSize = [imageWidth,imageHeight,imageDepth];
end

