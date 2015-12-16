cellDia = 5;
cellVol = (4*pi*cellDia^2);

montages = struct('filePath','','chanList',[]);
montages(end).filePath = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ CM#4 PumpSide 8-29-14\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x_Montage_wDelta\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x';
montages(end).chanList = [2];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ Control Pump-Side 9-2-14\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x_Montage_wDelta\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x';
montages(end).chanList = [2];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\SVZ CM-Pump 1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 10-31-14\SVZ CM Pump #1_Montage_wDelta\SVZ CM Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
montages(end).chanList = [2];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 11-4-14\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x_Montage_wDelta\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
montages(end).chanList = [2];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 11-14-14\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
montages(end).chanList = [2];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 11-18-14\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
montages(end).chanList = [2];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\22 Month Neg Control SVZ DAPI_DCX-488_GFAP-568_EDU-647 10-21-14\22 Month NegControl SVZ 10-21-14_Montage_wDelta\22 Month NegControl SVZ 10-21-14';
montages(end).chanList = [2];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 KD1 Deep Labels 7-21-13\DAPI Olig2-514 laminin-488 EdU-649 Nestin-Tomato PSA-NCAM-568 20x_Montage_wDelta\Itga9 KD1 Deep Labels 7-21-13';
montages(end).chanList = [4];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 kd2 Deep Labels 8-03-13\DAPI GFAP-514 laminin-488 EdU-647 Nestin--Tomato PSA-NCAM-594 kd2 _Montage_wDelta\Itga9 kd2 Deep Labels 8-03-13';
montages(end).chanList = [4];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 kd3(J4) Deep Labels 8-12-13 take2\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd2(J1) Surface Labels 8-19-13';
montages(end).chanList = [4];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 kd5(J3) Deep Labels 8-17-13\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd5(J3) Deep Labels 8-17-13';
montages(end).chanList = [4];

eduCount = zeros(length(montages),1);

for m=1:length(montages)
    [im,imD] = MicroscopeData.Tiff.Reader(montages(m).filePath, [], montages(m).chanList);
    imBW = false(size(im));
    
    curIm = im(:,:,:);
    fprintf('Threshold...');
    tic
    imBW(:,:,:) = curIm>255*graythresh(curIm(curIm>0));
    fprintf('done (%s)\n',Utils.PrintTime(toc))
    
    tic
    fprintf('cc...');
    cc = bwconncomp(imBW(:,:,:));
    fprintf('done (%s)\n',Utils.PrintTime(toc))
    
    tic
    fprintf('regionProps...');
    rp = regionprops(imBW(:,:,:),curIm,'Area','PixelList');
    fprintf('done (%s)\n',Utils.PrintTime(toc))
    
    cellVox = cellVol/(imD.XPixelPhysicalSize*imD.YPixelPhysicalSize*imD.ZPixelPhysicalSize);
    
    ccCur = cc.PixelIdxList;
    rpCur = rp;
    
    cells = cell(length(rpCur),1);
    tic
    fprintf('gap...');
    parfor i=1:cc{c}.NumObjects
        thisRP = rpCur(i);
        curIdxList = ccCur{i};
        
        maxK = ceil(1.25 * (thisRP.Area / cellVox));
        voxLabels = Gap.GetLabels(thisRP.PixelList,maxK,'ke',true,false);
        PixelIdxList = cell(max(voxLabels),1);
        for j=1:max(voxLabels)
            PixelIdxList{j} = curIdxList(voxLabels==j);
        end
        cells{i} = PixelIdxList;
    end
    fprintf('done (%s)\n',Utils.PrintTime(toc))
    
    ccNew.Connectivity = cc.Connectivity;
    ccNew.ImageSize = cc.ImageSize;
    ccNew.NumObjects = 0;
    ccNew.PixelIdxList = {};
    for i=1:length(cells)
        curCell = cells{i};
        for j=1:size(curCell,1)
            ccNew.NumObjects = ccNew.NumObjects +1;
            ccNew.PixelIdxList{end+1} = curCell{j,1};
        end
    end
    
    imL = zeros(size(imBW(:,:,:,2)));
    
    stats = regionProps(ccNew,'Area');
    numEdu = 0;
    
    for i=1:length(ccNew.PixelIdxList)
        idx = ccNew.PixelIdxList{i};
        imL(idx) = i;
        
        if (stats(i).Area > cellVox*0.25)
            numEdu = numEdu +1;
        end
    end
    
    ImUtils.ShowMaxImage(imL,1);
    colormap lines;
    map = colormap;
    map(1,:) = [0,0,0];
    colormap(map);
    
    eduCount(m) = numEdu;
end
