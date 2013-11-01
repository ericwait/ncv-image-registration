function createMetadata(root,datasetName,imageData)
fileHandle = fopen([root '\' datasetName '.txt'],'wt');
fprintf(fileHandle,'DatasetName:%s\n',datasetName);
fprintf(fileHandle,'NumberOfChannels:%d\n',imageData.NumberOfChannels);
fprintf(fileHandle,'NumberOfFrames:%d\n',imageData.NumberOfFrames);
fprintf(fileHandle,'XDimension:%d\n',imageData.XDimension);
fprintf(fileHandle,'YDimension:%d\n',imageData.YDimension);
fprintf(fileHandle,'ZDimension:%d\n',imageData.ZDimension);
fprintf(fileHandle,'XPixelPhysicalSize:%f\n',imageData.XPixelPhysicalSize);
fprintf(fileHandle,'YPixelPhysicalSize:%f\n',imageData.YPixelPhysicalSize);
fprintf(fileHandle,'ZPixelPhysicalSize:%f\n',imageData.ZPixelPhysicalSize);
if (isfield(imageData,'XPosition'))
    fprintf(fileHandle,'XPosition:%f\n',imageData.XPosition);
end
if (isfield(imageData,'YPosition'))
    fprintf(fileHandle,'YPosition:%f\n',imageData.YPosition);
end
fclose(fileHandle);
end
