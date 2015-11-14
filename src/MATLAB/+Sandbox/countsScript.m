% montages = {};
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ CM#4 PumpSide 8-29-14\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x_Montage_wDelta\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ Control Pump-Side 9-2-14\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x_Montage_wDelta\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\SVZ CM-Pump 1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 10-31-14\SVZ CM Pump #1_Montage_wDelta\SVZ CM Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 11-4-14\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x_Montage_wDelta\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 11-14-14\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 11-18-14\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\22 Month Neg Control SVZ DAPI_DCX-488_GFAP-568_EDU-647 10-21-14\22 Month NegControl SVZ 10-21-14_Montage_wDelta\22 Month NegControl SVZ 10-21-14';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 KD1 Deep Labels 7-21-13\DAPI Olig2-514 laminin-488 EdU-649 Nestin-Tomato PSA-NCAM-568 20x_Montage_wDelta\Itga9 KD1 Deep Labels 7-21-13';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 kd2 Deep Labels 8-03-13\DAPI GFAP-514 laminin-488 EdU-647 Nestin--Tomato PSA-NCAM-594 kd2 _Montage_wDelta\Itga9 kd2 Deep Labels 8-03-13';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 kd3(J4) Deep Labels 8-12-13 take2\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd2(J1) Surface Labels 8-19-13';
% montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 kd5(J3) Deep Labels 8-17-13\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd5(J3) Deep Labels 8-17-13';

%features = struct('numVox',{[0,0,0]},'voxVol',{[0,0,0]},'numCC',{[0,0,0]});%,'roiVol',0,'roiVox',0);
% features(length(montages)).numVox = [0,0,0];
% 
% %load('features.mat');
% 
for i=1:length(montages)
    tic
    curMontage = montages(i);
    
    imMeta = MicroscopeData.ReadMetadata(curMontage.filePath);
%     im = MicroscopeData.Reader(imMeta);
%     imBW = false(size(im));
%     for c=1:imMeta.NumberOfChannels
%         imC = im(:,:,:,c);
%         imBW(:,:,:,c) = imC>graythresh(imC(imC>0))*255;
%     end
%     %[imBW,imData] = MicroscopeData.Reader(fullfile(imMeta.imageDir,[imMeta.DatasetName, '_seg']),[],[],[],'logical');
%     [pathstr,name,ext] = fileparts(montages(i).filePath);
%     imMask = imread(fullfile(pathstr,['_',imMeta.DatasetName,'_RGB_mask.tif']));
%     imMask = repmat(imMask,1,1,imMeta.ZDimension);
%     imArea = false(size(im,1),size(im,2),size(im,3));
%     for c=1:size(imBW,4)
%         imArea = imArea | imBW(:,:,:,c);
%     end
%     imArea = imArea & imMask;
%     
%     features(i).SVZarea = sum(imArea(:));
features(i).vox = prod([imMeta.XPixelPhysicalSize,imMeta.YPixelPhysicalSize,imMeta.ZPixelPhysicalSize]);
    
%     voxVol = prod([imData.XPixelPhysicalSize,imData.YPixelPhysicalSize,imData.ZPixelPhysicalSize]);
%     expectedVol = 4/3*pi*5^3/voxVol;
% 
%     for c=1:imData.NumberOfChannels
%         rp = regionprops(imBW(:,:,:,c),'Area');
%         features(i).numVox(c) = sum([rp(:).Area]);
%         features(i).voxVol(c) = features(i).numVox(c)/voxVol;
%         if (c==1 || c==2)
%             areas = [rp.Area];
%             bigIdx = find(areas>expectedVol);
%             numCC = length(rp) - length(bigIdx);
%             bigAreas = areas(bigIdx);
%             splitAreas = round(bigAreas./expectedVol);
%             numCC = numCC + sum(splitAreas);
%             features(i).numCC(c) = numCC;
%         end
%     end
%     
%     clear imBW
%     
%     [im,imFullData] = MicroscopeData.Reader(montages(i).filePath,[],montages(i).chanList);
%     colors = MicroscopeData.GetChannelColors(imFullData);
%     chans = curMontage.chanList(2:end);
%     imFinal = ImUtils.ThreeD.ColorMIP(im(:,:,:,2:end),colors(chans,:));
%     imwrite(imFinal,fullfile(montages(i).filePath,sprintf('_%s_%dc.tif',imFullData.DatasetName,size(im,4))),'Compression','lzw');
%     
%     clear im
%     clear imFinal

    Utils.PrintTime(toc)
end

save('features.mat','features');

i = 1;

datasetNames = {};

for i=1:length(montages)
    imMeta = MicroscopeData.ReadMetadata(montages(i).filePath);

    datasetNames{i} = imMeta.DatasetName;
end

figure
eduVoxAx = gca;
hold on

for i=1:length(montages)
    x = features(i).numVox(2);%/features(i).SVZarea;
    y = features(i).numVox(3);%/features(i).SVZarea;
plot(eduVoxAx,x,y,...
    'MarkerFaceColor',montages(i).faceColor,...
    'marker',montages(i).marker,...
    'MarkerSize',20);

text(x,y,...
    num2str(i),'parent',eduVoxAx,'color',xor([1,1,1],montages(i).faceColor));

end

title('Channel Area');
xlabel('EdU Area');% over SVZ area');
ylabel('Dcx Area');% SVZ area');

legend(datasetNames,'location','northeastoutside')

