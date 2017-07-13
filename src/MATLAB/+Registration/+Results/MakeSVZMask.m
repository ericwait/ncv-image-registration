function MakeSVZMask(im,imD,maxSide)
if (~exist('imD','var') || isempty(imD))
    imD = MicroscopeData.ReadMetadata();
end

disp(imD.DatasetName)
if (~exist('im','var') || isempty(im))
    im = MicroscopeData.ReaderH5('imageData',imD,'imVersion','Processed','getMIP',true);
end
colors = MicroscopeData.Colors.GetChannelColors(imD);

imRot = Helper.GetAxisOffsetAngle(imD);

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

bwRot = imrotate(bw,imRot);
if (~exist('maxSide','var') || isempty(maxSide))
    reducAmnt = 1;
else
    reducAmnt = maxSide/size(bwRot,2);
end

imSmall = ImProc.Resize(im2uint8(bwRot),[reducAmnt,reducAmnt,1],[],'mean')>128;
pixlist_rc = Utils.IndToCoord(size(imSmall),find(imSmall));
bb = [min(pixlist_rc,[],1);max(pixlist_rc,[],1)];
bb(1,:) = max([1,1],bb(1,:)-150);
bb(2,:) = min(size(imSmall),bb(2,:)+150);

imSmall = imSmall(bb(1,1):bb(2,1),bb(1,2):bb(2,2));
    
%% get the boundry pixels dialated a bit
bound = bwperim(imSmall);
se = strel('disk',5,4);
bound = imdilate(bound,se);
boundInd = bound>0;
    
%% get the number of image combinations we want
% masks = zeros(2^imD.NumberOfChannels,imD.NumberOfChannels);
% for i=1:2^imD.NumberOfChannels
%     masks(i,:) = bitget(i,imD.NumberOfChannels:-1:1);
% end
% masks = masks>0;

% prgs = Utils.CmdlnProgress(size(masks,1),false,'Making Color Combinations');
chans = 1:imD.NumberOfChannels;
for j=63%1:size(masks,1)
    curChans = chans;%(masks(j,:));
    if (isempty(curChans))
        continue
    end
    
    imageName = sprintf('_%s_chan%s.tif',imD.DatasetName,num2str(curChans,'%d'));
    imagePath = fullfile(imD.imageDir,imageName);
    
    curIm = im(:,:,:,curChans);
    colorMip = ImUtils.ThreeD.ColorMIP(curIm,colors(curChans,:));
    imwrite(colorMip,imagePath,'tif','Compression','lzw');
    
    colorMip = imrotate(colorMip,imRot);
    colorMip = ImProc.Resize(colorMip,[reducAmnt,reducAmnt,1],[],'mean');
    colorMip = colorMip(bb(1,1):bb(2,1),bb(1,2):bb(2,2),:);
    colorMip = ImUtils.ROI.GetMaskedIm(colorMip,imSmall,boundInd);
    imwrite(colorMip,imagePath,'tif','Compression','lzw');
    
    fprintf('Wrote: %s\n',imageName);
    %prgs.PrintProgress(j);
end

%prgs.ClearProgress(true);

end
