load('montages.mat');

features = struct('numVox',{[0,0,0]},'voxVol',{[0,0,0]},'numCC',{[0,0,0]},'voxScale',{0});
features(length(montages)).numVox = [0,0,0];

for i=1:length(montages)
    tic
    
    %% read in segmentation images
    curMontage = montages(i);
    imDataOrg = MicroscopeData.ReadMetadata(curMontage.filePath);
    [imBW,imDataBW] = MicroscopeData.Reader(fullfile(imDataOrg.imageDir,[imDataOrg.DatasetName, '_seg']),[],[],[],'logical');
    
    %% read in ROI mask
    [pathstr,name,ext] = fileparts(montages(i).filePath);
    maskFile = fullfile(pathstr,['_',imDataOrg.DatasetName,'_mask.tif']);
    imMask = imread(maskFile);
    imMask = repmat(imMask,1,1,imDataOrg.ZDimension);
    
    %% find the area of the ROI given the segmentation
    imArea = false(imDataOrg.YDimension,imDataOrg.XDimension,imDataOrg.ZDimension);
    for c=1:size(imBW,4)
        imArea = imArea | imBW(:,:,:,c);
    end
    imArea = imArea & imMask;
    features(i).SVZarea = sum(imArea(:));
    
    %% Get features for each channel
    voxVol = prod([imDataBW.XPixelPhysicalSize,imDataBW.YPixelPhysicalSize,imDataBW.ZPixelPhysicalSize]);
    expectedVol = 4/3*pi*5^3/voxVol;
    
    features(i).voxScale = voxVol;

    for c=1:imDataBW.NumberOfChannels
        rp = regionprops(imBW(:,:,:,c),'Area','PixelIdxList');
        areas = [rp.Area];
        littleIdx = areas<expectedVol*0.5;
        features(i).numVox(c) = sum(cellfun(@(x) length(x),{rp(~littleIdx).PixelIdxList}));
        features(i).voxVol(c) = sum([rp(~littleIdx).Area])/voxVol;
        features(i).numCC(c) = sum(~littleIdx);
        
        if (c~=3)
            % get number of connected components that are bigger then the
            % expected size
            bigIdx = areas>expectedVol*1.5;
            numCC = sum(~littleIdx & ~bigIdx);
            
            % split up each connected component by volume
            bigAreas = areas(bigIdx);
            splitAreas = round(bigAreas./expectedVol);
            numCC = numCC + sum(splitAreas);
            features(i).numCC(c) = numCC;
        end
    end
    
    clear imBW

    Utils.PrintTime(toc)
end

save('features.mat','features');
