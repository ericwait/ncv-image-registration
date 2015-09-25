montages = {};
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ CM#4 PumpSide 8-29-14\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x_Montage_wDelta\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ Control Pump-Side 9-2-14\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x_Montage_wDelta\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\SVZ CM-Pump 1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 10-31-14\SVZ CM Pump #1_Montage_wDelta\SVZ CM Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 11-4-14\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x_Montage_wDelta\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 11-14-14\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 11-18-14\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\22 Month Neg Control SVZ DAPI_DCX-488_GFAP-568_EDU-647 10-21-14\22 Month NegControl SVZ 10-21-14_Montage_wDelta\22 Month NegControl SVZ 10-21-14';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 KD1 Deep Labels 7-21-13\DAPI Olig2-514 laminin-488 EdU-649 Nestin-Tomato PSA-NCAM-568 20x_Montage_wDelta\Itga9 KD1 Deep Labels 7-21-13';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 kd2 Deep Labels 8-03-13\DAPI GFAP-514 laminin-488 EdU-647 Nestin--Tomato PSA-NCAM-594 kd2 _Montage_wDelta\Itga9 kd2 Deep Labels 8-03-13';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 kd3(J4) Deep Labels 8-12-13 take2\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd2(J1) Surface Labels 8-19-13';
montages{end+1} = 'C:\Images\Temple\SVZ\Montage\Itga9 kd5(J3) Deep Labels 8-17-13\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd5(J3) Deep Labels 8-17-13';

% features = struct('numVox',[0,0,0],'voxVol',[0,0,0]);%,'roiVol',0,'roiVox',0);
% features(length(montages)).numVox = [0,0,0];
% 
% %load('features.mat');
% 
for i=9:length(montages)
    tic
    curMontage = montages{i};
%     
%     [imBW,imData] = tiffReader([curMontage, '_seg'],[],[],[],'logical');
% 
%     for c=1:imData.NumberOfChannels
%         rp = regionprops(imBW(:,:,:,c),'Area','Eccentricity');
%         features(i).numVox(c) = sum([rp(:).Area]);
%     end
    
    [pathstr,name,ext] = fileparts(curMontage);
    
        
    [imFinal,~] = colorMIP(curMontage);
    imwrite(imFinal,fullfile(pathstr,['_',name,'.tif']),'Compression','lzw');
    
    [im,imData] = tiffReader(curMontage,[],4);
    imFinal = zeros(size(im,1),size(im,2),3,'like',im);
    imFinal(:,:,1) = max(im(:,:,:),[],3);
    imwrite(imFinal,fullfile(pathstr,['_',name,'_c4.tif']),'Compression','lzw');
    printTime(toc)
end
% 
% save('features.mat','features');
