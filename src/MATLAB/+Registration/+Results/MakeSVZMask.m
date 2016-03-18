function MakeSVZMask()
imD = MicroscopeData.ReadMetadata();
disp(imD.DatasetName)

%% get the iamge data
mipfilePath = fullfile(imD.imageDir,[imD.DatasetName,'_MIP.tif']);
if (~exist(mipfilePath,'file'))
    mipfilePath = fullfile(imD.imageDir,['_',imD.DatasetName,'_MIP.tif']);
    if (~exist(mipfilePath,'file'))
        error('Cannot find file %s',mipfilePath);
    end
end
    
colorMip = imread(mipfilePath);
colors = MicroscopeData.Colors.GetChannelColors(imD);
    
bw = roipoly(colorMip);

imwrite(im2uint8(bw),fullfile(imD.imageDir,['_',imD.DatasetName,'_Mask.tif']),'compression','lzw');
    
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
    
    im = MicroscopeData.ReaderParZ(imD,[],curChans);
    colorMip = ImUtils.ThreeD.ColorMIP(im,colors(curChans,:));
    imwrite(colorMip,imagePath,'tif','Compression','lzw');
    
    colorMip = ImUtils.ROI.GetMaskedIm(colorMip,bw,boundInd);
    imwrite(colorMip,imagePath,'tif','Compression','lzw');
    
    fprintf('Wrote: %s\n',imageName);
    prgs.PrintProgress(j);
end

prgs.ClearProgress(true);

end
