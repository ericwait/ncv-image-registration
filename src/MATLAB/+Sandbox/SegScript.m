montages = struct('filePath','','chanList',[]);
montages(end).filePath = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ CM#4 PumpSide 8-29-14\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x_Montage_wDelta\DAPI EdU-647 Dcx-488 GFAP-Cy3 CM_Pump4ipsilateral 20x2x';
montages(end).chanList = [1,2,3];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\ChP-CM Pump wmSVZ Control Pump-Side 9-2-14\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x_Montage_wDelta\DAPI_DCX-488_GFAP-546_EDU-647 ChP-CM Pump 20x02x';
montages(end).chanList = [1,2,3];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\SVZ CM-Pump 1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 10-31-14\SVZ CM Pump #1_Montage_wDelta\SVZ CM Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
montages(end).chanList = [1,2,3];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 11-4-14\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x_Montage_wDelta\SVZ Control Pump #1 DAPI EDU-647 DCX- 488 GFAP-568 Pumpside 20x02x';
montages(end).chanList = [1,2,3];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 11-14-14\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ CM Pump#2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
montages(end).chanList = [1,2,3];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 11-18-14\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x_Montage_wDelta\wmSVZ Control Pump #2 DAPI_EDU-647_DCX-488_GFAP-568 20x02x';
montages(end).chanList = [1,2,3];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\22 Month Neg Control SVZ DAPI_DCX-488_GFAP-568_EDU-647 10-21-14\22 Month NegControl SVZ 10-21-14_Montage_wDelta\22 Month NegControl SVZ 10-21-14';
montages(end).chanList = [1,2,3];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 KD1 Deep Labels 7-21-13\DAPI Olig2-514 laminin-488 EdU-649 Nestin-Tomato PSA-NCAM-568 20x_Montage_wDelta\Itga9 KD1 Deep Labels 7-21-13';
montages(end).chanList = [1,4];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 kd2 Deep Labels 8-03-13\DAPI GFAP-514 laminin-488 EdU-647 Nestin--Tomato PSA-NCAM-594 kd2 _Montage_wDelta\Itga9 kd2 Deep Labels 8-03-13';
montages(end).chanList = [1,4];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 kd3(J4) Deep Labels 8-12-13 take2\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd2(J1) Surface Labels 8-19-13';
montages(end).chanList = [1,4];

montages(end+1).filePath = 'C:\Images\Temple\SVZ\Montage\Itga9 kd5(J3) Deep Labels 8-17-13\DAPI GFAP-514 laminin-488 EdU-647 NestinTomato PSA-NCAM-594 20x_Montage_wDelta\Itga9 kd5(J3) Deep Labels 8-17-13';
montages(end).chanList = [1,4];


for i=1:length(montages)
    fullStart = tic;
    readStart = tic;
    [pathstr,name,ext] = fileparts(montages(i).filePath);
    imData = readMetadata(montages(i).filePath);
    %if (~exist(fullfile(pathstr,[imData.DatasetName,'_seg.json']),'file'))
        disp([fullfile(pathstr,['_',imData.DatasetName,'_mask.tif']),'\n']);
        imMask = imread(fullfile(pathstr,['_',imData.DatasetName,'_mask.tif']));
        imMask = repmat(imMask,1,1,imData.ZDimension);
        imBW = false(imData.YDimension,imData.XDimension,imData.ZDimension,length(montages(i).chanList));
        %imBw = tiffReader(fullfile(pathstr,[imData.DatasetName,'_seg.json']),[],[],[],'logical');
        imChans = tiffReader(montages(i).filePath,[],montages(i).chanList);
        readTime = toc(readStart);
        [numBits,minVal,maxVal] = classBits(imChans);
        
        processStart = tic;
        for c=1:length(montages(i).chanList)
            im = imChans(:,:,:,c);
            im(~imMask) = 0;
            im = FluorescentBGRemoval(im);
            
            %do a two class threshold for the third channel
            if (c==3)
                th = graythresh(im(im>0));
                [counts,binCenters] = imhist(im(im>th*maxVal*0.8));
                thr = BackgroundThresh(counts,binCenters,th*maxVal*5,[],50);
                im(im<thr) = 0;
            end
            
            im = CudaMex('MedianFilter',im,[3,3,3]);
            im(~imMask) = 0;
            th = graythresh(im(im>0));
            imBW(:,:,:,c) = im>=(th*maxVal);
        end
        processTime = toc(processStart);
        
        writeStart = tic;
        imData.DatasetName = [imData.DatasetName,'_seg'];
        imData.NumberOfChannels = length(montages(i).chanList);
        tiffWriter(imBW,pathstr,imData);
        writeTime = toc(writeStart);
        fullTime = toc(fullStart);
        
        fprintf('TotalTime:%s, read:%s, process:%s, write:%s\n',...
            printTime(fullTime),printTime(readTime),printTime(processTime),printTime(writeTime));
    %end
end

clear imMask
clear imBW
clear im
