function MakeSVZMask(im,imD)
if (~exist('imD','var') || isempty(imD))
    imD = MicroscopeData.ReadMetadata();
end

disp(imD.DatasetName)
if (~exist('im','var') || isempty(im))
    im = MicroscopeData.ReaderH5('imageData',imD);
end
colors = MicroscopeData.Colors.GetChannelColors(imD);

%% get the iamge data
maskPath = fullfile(imD.imageDir,['_',imD.DatasetName,'_Mask.tif']);
if (~exist(maskPath,'file'))
    suffix = '_chan';
    for c=2:imD.NumberOfChannels
        suffix = [suffix,num2str(c)];
    end
    suffix = [suffix,'.tif'];
    
    mipfilePath = fullfile(imD.imageDir,['_',imD.DatasetName,suffix]);
    if (~exist(mipfilePath,'file'))
        mipfilePath = fullfile(imD.imageDir,['_',imD.DatasetName,'_MIP.tif']);
        if (~exist(mipfilePath,'file'))
            colorMip = ImUtils.ThreeD.ColorMIP(im,colors);
        else
            colorMip = imread(mipfilePath);
        end
    else
        colorMip = imread(mipfilePath);
    end
    
    bw = roipoly(colorMip);
    imwrite(im2uint8(bw),maskPath,'compression','lzw');
else
    bw = imread(maskPath);
end
    
%% get the boundry pixels dialated a bit
bound = bwperim(bw);
se = strel('disk',5,4);
bound = imdilate(bound,se);
boundInd = bound>0;
    
%% get the number of image combinations we want
masks = zeros(2^imD.NumberOfChannels,imD.NumberOfChannels);
for i=1:2^imD.NumberOfChannels
    masks(i,:) = bitget(i,imD.NumberOfChannels:-1:1);
end
masks = masks>0;

prgs = Utils.CmdlnProgress(size(masks,1),false,'Making Color Combinations');
chans = 1:imD.NumberOfChannels;
for j=1:size(masks,1)
    curChans = chans(masks(j,:));
    if (isempty(curChans))
        continue
    end
    
    imageName = sprintf('_%s_chan%s.tif',imD.DatasetName,num2str(curChans,'%d'));
    imagePath = fullfile(imD.imageDir,imageName);
    
    curIm = im(:,:,:,curChans);
    colorMip = ImUtils.ThreeD.ColorMIP(curIm,colors(curChans,:));
    imwrite(colorMip,imagePath,'tif','Compression','lzw');
    
    colorMip = ImUtils.ROI.GetMaskedIm(colorMip,bw,boundInd);
    imwrite(colorMip,imagePath,'tif','Compression','lzw');
    
    fprintf('Wrote: %s\n',imageName);
    prgs.PrintProgress(j);
end

prgs.ClearProgress(true);

end
