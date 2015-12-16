imData.NumberOfChannels = 2;
imData.ZDimension = 1;
imData.NumberOfFrames = 1;
imageDataset = imData;

minOverlap = 10;
percentOverlap = 0.10;
overlapInit = [ceil(imageDataset.YDimension * percentOverlap),ceil(imageDataset.XDimension * percentOverlap)];
maxSearchSize = max(overlapInit)*0.5;

for r=1:10
    imageDataset.XPosition = ((r-1)*imageDataset.YDimension-(r-1)*overlapInit(1))*1e-6;
    
    for c=1:10
        imageDataset.YPosition = ((c-1)*imageDataset.XDimension-(c-1)*overlapInit(2))*1e-6;
        
        imageDataset.DatasetName = sprintf('%s-R%02dC%02d',imData.DatasetName,r,c);
        sz = size(im(:,:,:,r,c));
        imTemp = zeros([sz(1),sz(2),1,sz(3)],'like',im);
        imTemp(:,:,1,:) = im(:,:,:,r,c);
        
        tiffWriter(imTemp,fullfile(imData.imageDir,imageDataset.DatasetName),imageDataset);
    end
end