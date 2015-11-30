function MakeSVZMask(montages,redoAll,prompt)

if (~exist('redoAll','var') || isempty(redoAll))
    redoAll = false;
end
if (~exist('prompt','var') || isempty(prompt))
    prompt = false;
end

for i=1:length(montages)
    %% get the iamge data
    [im,imageData] = MicroscopeData.Reader(montages(i).filePath);
    colors = MicroscopeData.GetChannelColors(imageData);
    colorMip = ImUtils.ThreeD.ColorMIP(im,colors);
    
    maskPath = fullfile(montages(i).filePath,['_',imageData.DatasetName, '_mask.tif']);
    
    %% read the mask image or create it
    if (exist(maskPath,'file') && ~redoAll)
        if (prompt)
            rspnce = questdlg('Would you like to redraw?','Redo ROI','Yes','No','No');
            if (strcmpi('yes',rspnce))
                bw = roipoly(colorMip);
                if (isempty(bw))
                    continue
                end
            else
                bw = imread(maskPath);
            end
        else
            bw = imread(maskPath);
        end
    else
        bw = roipoly(colorMip);
    end
    
    %% get the boundry pixels dialated a bit
    bound = bwperim(bw);
    se = strel('disk',5,4);
    bound = imdilate(bound,se);
    boundInd = bound>0;
    
    %% get the number of image combinations we want
    masks = Utils.MakeBinaryMasks(imageData.NumberOfChannels);
    masks(1:end-1,1) = false(numel(1:size(masks,1)-1),1);
    masks = unique(masks,'rows');
    
    prgs = Utils.CmdlnProgress(size(masks,1),false);
    chans = 1:imageData.NumberOfChannels;
    for j=1:size(masks,1)
        curChans = chans(masks(j,:));
        if (isempty(curChans))
            continue
        end
        
        imageName = sprintf('_%s_chan%s.tif',imageData.DatasetName,num2str(curChans,'%d'));
        imagePath = fullfile(montages(i).filePath,imageName);
        imageNameMask = sprintf('_%s_masked_chan%s.tif',imageData.DatasetName,num2str(curChans,'%d'));
        imagePathMask = fullfile(montages(i).filePath,imageNameMask);
        
        if (~exist(imagePath,'file') || ~exist(imagePathMask,'file') || redoAll)
            colorMip = ImUtils.ThreeD.ColorMIP(im(:,:,:,curChans),colors(curChans,:));            
            imwrite(colorMip,imagePath,'tif','Compression','lzw');
            
            colorMip = ImUtils.ROI.GetMaskedIm(colorMip,bw,boundInd);
            imwrite(colorMip,imagePath,'tif','Compression','lzw');
            
            fprintf('Wrote: %s\n',imageName);
        end
        prgs.PrintProgress(j);
    end
    
    prgs.ClearProgress();
end
end
