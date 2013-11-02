function createMetadata(root,datasetName,imageData)
fileHandle = fopen([root '\' datasetName '.txt'],'wt');
fprintf(fileHandle,'DatasetName:%s\n',datasetName);
fprintf(fileHandle,'NumberOfChannels:%d\n',imageData.NumberOfChannels);
fprintf(fileHandle,'NumberOfFrames:%d\n',imageData.NumberOfFrames);
fprintf(fileHandle,'XDimension:%d\n',imageData.xDim);
fprintf(fileHandle,'YDimension:%d\n',imageData.yDim);
fprintf(fileHandle,'ZDimension:%d\n',imageData.zDim);
fprintf(fileHandle,'XPixelPhysicalSize:%f\n',imageData.xVoxelSize);
fprintf(fileHandle,'YPixelPhysicalSize:%f\n',imageData.yVoxelSize);
fprintf(fileHandle,'ZPixelPhysicalSize:%f\n',imageData.zVoxelSize);
if (isfield(imageData,'XPosition'))
    fprintf(fileHandle,'XPosition:%f\n',imageData.XPosition);
end
if (isfield(imageData,'YPosition'))
    fprintf(fileHandle,'YPosition:%f\n',imageData.YPosition);
end
fclose(fileHandle);
end