function createMetadata(root,imageData)
fileName = fullfile(root,[imageData.DatasetName '.txt']);
fprintf('Creating Metadata %s...',fileName);

fileHandle = fopen(fileName,'wt');

fprintf(fileHandle,'DatasetName:%s\n',imageData.DatasetName);

fprintf(fileHandle,'NumberOfChannels:%d\n',imageData.NumberOfChannels);

if (isfield(imageData,'ChannelColors'))
    fprintf(fileHandle,'ChannelColors:');
    for i=1:length(imageData.ChannelColors)
        fprintf(fileHandle,'%s,',imageData.ChannelColors{i});
    end
    fprintf(fileHandle,'\n');
end

fprintf(fileHandle,'NumberOfFrames:%d\n',imageData.NumberOfFrames);

fprintf(fileHandle,'XDimension:%d\n',imageData.xDim);
fprintf(fileHandle,'YDimension:%d\n',imageData.yDim);
fprintf(fileHandle,'ZDimension:%d\n',imageData.zDim);

fprintf(fileHandle,'XPixelPhysicalSize:%f\n',imageData.XPixelPhysicalSize);
fprintf(fileHandle,'YPixelPhysicalSize:%f\n',imageData.YPixelPhysicalSize);
fprintf(fileHandle,'ZPixelPhysicalSize:%f\n',imageData.ZPixelPhysicalSize);

if (isfield(imageData,'XPosition'))
    fprintf(fileHandle,'XPosition:%f\n',imageData.XPosition);
end
if (isfield(imageData,'YPosition'))
    fprintf(fileHandle,'YPosition:%f\n',imageData.YPosition);
end

if (isfield(imageData,'XDistanceUnits'))
    fprintf(fileHandle,'XDistanceUnits:%s\n',imageData.XDistanceUnits);
end
if (isfield(imageData,'YDistanceUnits'))
    fprintf(fileHandle,'YDistanceUnits:%s\n',imageData.YDistanceUnits);
end
if (isfield(imageData,'ZDistanceUnits'))
    fprintf(fileHandle,'ZDistanceUnits:%s\n',imageData.ZDistanceUnits);
end

if (isfield(imageData,'XLength'))
    fprintf(fileHandle,'XLength:%f\n',imageData.XLength);
end
if (isfield(imageData,'YLength'))
    fprintf(fileHandle,'YLength:%f\n',imageData.YLength);
end
if (isfield(imageData,'ZLength'))
    fprintf(fileHandle,'ZLength:%f\n',imageData.ZLength);
end
fclose(fileHandle);

fprintf('Done\n');
end