function [imFinal,metadataFileName] = colorMIP(metadataFilePath,chanList)
%% get image data
if (~exist('metadataFilePath','var'))
    metadataFilePath = [];
end

imageData = readMetadata(metadataFilePath);
metadataFilePath = fullfile(imageData.imageDir,imageData.DatasetName);

if (~exist('chanList','var') || isempty(chanList))
    chanList = 1:imageData.NumberOfChannels;
end

if ~isfield(imageData,'DatasetName'), return, end

tmpColors = GetChannelColors(imageData);

colors = zeros(1,1,3,length(chanList));
for c=1:length(chanList)
    colors(1,1,:,c) = tmpColors(chanList(c),:);
end

%% make colored image
imColors = zeros(imageData.YDimension,imageData.XDimension,3,length(chanList));
imIntensity = zeros(imageData.YDimension,imageData.XDimension,length(chanList));
for c=1:length(chanList)
    imIntensity(:,:,c) = mat2gray(max(tiffReader(metadataFilePath,[],chanList(c),[],[],false,1),[],3));
    color = repmat(colors(1,1,:,c),imageData.YDimension,imageData.XDimension,1);
    imColors(:,:,:,c) = repmat(imIntensity(:,:,c),1,1,3).*color;
end

imMax = max(imIntensity,[],3);
imIntSum = sum(imIntensity,3);
imIntSum(imIntSum==0) = 1;
imColrSum = sum(imColors,4);
imFinal = imColrSum.*repmat(imMax./imIntSum,1,1,3);
imFinal = im2uint8(imFinal);
[root,~,~] = fileparts(metadataFilePath);
metadataFileName = fullfile(root,sprintf('_%s.tif',imageData.DatasetName));
end
