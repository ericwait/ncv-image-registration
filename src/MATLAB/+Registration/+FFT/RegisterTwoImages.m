function [ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNCV,overlapSize] = RegisterTwoImages(im1,imageDataset1,im2,...
    imageDataset2,unitFactor,minOverlap,maxSearchSize,logFile,visualize, imMask1, imMask2)
clear global Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar MaxCovar SubImBest1 SubImBest2 DecisionFig DecisionAxes
global Rect1 Rect2

%% check inputs
if (~exist('minOverlap','var') || isempty(minOverlap))
    minOverlap = 25;
end
if (~exist('maxSearchSize','var') || isempty(maxSearchSize))
    maxSearchSize = 100;
end
if (~exist('logFile','var'))
    logFile = 1;
end
if (~exist('visualize','var') || isempty(visualize))
    visualize = 0;
end
if (~exist('imMask1','var'))
    imMask1 = [];
end
if (~exist('imMask2','var'))
    imMask2 = [];
end

if (~exist('imageDataset1','var') || isempty(imageDataset1))
    imageDataset1 = MicroscopeData.GetEmptyMetadata();
    sz = [size(im1,1),size(im1,2),size(im1,3),size(im1,4),size(im1,5)];
    imageDataset1.Dimensions = sz([2,1,3]);
    imageDataset1.NumberOfChannels = sz(4);
    imageDataset1.NumberOfFrames = sz(5);
    imageDataset1.DatasetName = 'image 1';
    imageDataset1.PixelPhysicalSize = ones(1,3);
end

if (~exist('imageDataset2','var') || isempty(imageDataset2))
    imageDataset2 = MicroscopeData.GetEmptyMetadata();
    sz = [size(im2,1),size(im2,2),size(im2,3),size(im2,4),size(im2,5)];
    imageDataset2.Dimensions = sz([2,1,3]);
    imageDataset2.NumberOfChannels = sz(4);
    imageDataset2.NumberOfFrames = sz(5);
    imageDataset2.DatasetName = 'image 1';
    imageDataset2.PixelPhysicalSize = ones(1,3);
end

if (exist('logFile','var') && ~isempty(logFile))
    if (logFile~=1)
        fHand = fopen(logFile,'at');
    else
        fHand = 1;
    end
    fprintf(fHand,'%s \n\t--> %s\n',imageDataset1.DatasetName,imageDataset2.DatasetName);
    if (fHand~=1)
        fclose(fHand);
    end
end

%% check to see if the image has data and is big enough
[imageROI1Org_XY,imageROI2Org_XY,~,~] = Registration.Overlap.CalculateOverlapXY(imageDataset1,imageDataset2,unitFactor);
[imageROI1_XY,imageROI2_XY,padding_XY] = Registration.Overlap.AddPaddingToOverlapXY(imageDataset1,imageROI1Org_XY,imageDataset2,imageROI2Org_XY,maxSearchSize);

maxNCV = -inf;
bestChan = 0;
ultimateDeltaX = 0;
ultimateDeltaY = 0;
ultimateDeltaZ = 0;

overlapSize = max(min(imageROI1Org_XY(4)-imageROI1Org_XY(1),imageROI2Org_XY(4)-imageROI2Org_XY(1)),1) *...
    max(min(imageROI1Org_XY(5)-imageROI1Org_XY(2),imageROI2Org_XY(5)-imageROI2Org_XY(2)),1);

im1ROI = im1(imageROI1_XY(2):imageROI1_XY(5),imageROI1_XY(1):imageROI1_XY(4),imageROI1_XY(3):imageROI1_XY(6),:,:);
im2ROI = im2(imageROI2_XY(2):imageROI2_XY(5),imageROI2_XY(1):imageROI2_XY(4),imageROI2_XY(3):imageROI2_XY(6),:,:);
if (~isempty(imMask1))
    imMask1ROI = imMask1(imageROI1_XY(2):imageROI1_XY(5),imageROI1_XY(1):imageROI1_XY(4),imageROI1_XY(3):imageROI1_XY(6),:,:);
else
    imMask1ROI = [];
end
if (~isempty(imMask2))
    imMask2ROI = imMask2(imageROI2_XY(2):imageROI2_XY(5),imageROI2_XY(1):imageROI2_XY(4),imageROI2_XY(3):imageROI2_XY(6),:,:);
else
    imMask2ROI = [];
end

[~,~,maxVal] = Utils.GetClassBits(im1ROI);
% if (max(im1ROI(:))<=0.28*maxVal || max(im2ROI(:))<=0.28*maxVal)
%     % no real info in the image
%     return
% end
if (overlapSize <= 0.01* min(prod(imageDataset1.Dimensions(1:2)),prod(imageDataset2.Dimensions(1:2))))
    % does not have enough overall overlap
    warning('No overlap found');
    return
end

%% run 2-D case
newOrgin_RC = Utils.SwapXY_RC(padding_XY);
totalTm = tic;

im1MaxROI = squeeze(max(im1ROI,[],3));
im2MaxROI = squeeze(max(im2ROI,[],3));
for c=1:imageDataset1.NumberOfChannels
    if (visualize==true)
        im1MaxChan = squeeze(max(im1(:,:,:,c),[],3));
        im2MaxChan = squeeze(max(im2(:,:,:,c),[],3));
        Registration.Iterative.SetupVisualizer(im1MaxChan,im2MaxChan,imageROI1Org_XY,imageROI2Org_XY,imageDataset1,imageDataset2);
    end
    
    [deltas_RC,curNCV] = Registration.FFT.GetMaxNCVdeltas(im1MaxROI(:,:,c),im2MaxROI(:,:,c),minOverlap^2,maxSearchSize,newOrgin_RC([1,2]),visualize,c,[],imMask1ROI, imMask2ROI);
    deltas_XY = Utils.SwapXY_RC(deltas_RC);
    if (curNCV>maxNCV)
        bestChan = c;
        maxNCV = curNCV;
        bestDeltas_XY = deltas_XY;
    end
    
    if (visualize==true)
        [xStart1,xStart2,xEnd1,xEnd2] = Registration.Overlap.CalculateROIs(deltas_XY(1),imageROI1Org_XY(1),imageROI2Org_XY(1),size(im1,2),size(im2,2));
        [yStart1,yStart2,yEnd1,yEnd2] = Registration.Overlap.CalculateROIs(deltas_XY(2),imageROI1Org_XY(2),imageROI2Org_XY(2),size(im1,1),size(im2,1));

        set(Rect1,'Position',[max(xStart1,1),max(yStart1,1),max(xEnd1-xStart1,1),max(yEnd1-yStart1,1)]);
        set(Rect2,'Position',[max(xStart2,1),max(yStart2,1),max(xEnd2-xStart2,1),max(yEnd2-yStart2,1)]);
        imY1 = im1MaxChan(yStart1:yEnd1,xStart1:xEnd1);
        imY2 = im2MaxChan(yStart2:yEnd2,xStart2:xEnd2);

        Registration.Iterative.UpdateXYviewer(imY1,imY2,curNCV,deltas_XY(1),deltas_XY(2),deltas_XY(3));
    end
end

tm = toc(totalTm);
if (~isempty(logFile))
    if (logFile~=1)
        fHand = fopen(logFile,'at');
    else
        fHand = 1;
    end
    fprintf(fHand,'\t%s, NVC:%04.3f at (%d,%d) on channel:%d\n',...
        Utils.PrintTime(tm),maxNCV,bestDeltas_XY(1),bestDeltas_XY(2),bestChan);
    if (fHand~=1)
        fclose(fHand);
    end
end

bestDeltas_XY = [bestDeltas_XY([1,2]),0];
deltasZ_XY = bestDeltas_XY;
maxNcovZ = maxNCV;

%% run 3-D case
if (size(im1,3)>1)
    totalTm = tic;
    
    if (visualize==true)
        im1MaxChan = squeeze(max(im1(:,:,:,bestChan),[],3));
        im2MaxChan = squeeze(max(im2(:,:,:,bestChan),[],3));
        Registration.Iterative.SetupVisualizer(im1MaxChan,im2MaxChan,imageROI1Org_XY,imageROI2Org_XY,imageDataset1,imageDataset2);
    end
    
    [deltasZ_RC,maxNcovZ] = Registration.FFT.GetMaxNCVdeltas(im1ROI(:,:,:,bestChan),im2ROI(:,:,:,bestChan),minOverlap^3,maxSearchSize,newOrgin_RC,visualize,bestChan);
    deltasZ_XY = Utils.SwapXY_RC(deltasZ_RC);
    
    tm = toc(totalTm);
   
    changeDelta_XY = bestDeltas_XY - deltasZ_XY;
    
    if (~isempty(logFile))
        if (logFile~=1)
            fHand = fopen(logFile,'at');
        else
            fHand = 1;
        end
        
        fprintf(fHand,'\t%s, NVC:%04.3f at (%d,%d,%d)\n',...
            Utils.PrintTime(tm),maxNcovZ,deltasZ_XY(1),deltasZ_XY(2),deltasZ_XY(3));
        
        if (changeDelta_XY(1)~=0 || changeDelta_XY(2)~=0)
            fprintf(fHand,'\tA better delta was found when looking in Z. Change in deltas=(%d,%d,%d) Old NCV:%f new:%f\n', changeDelta_XY(1),changeDelta_XY(2),changeDelta_XY(3),maxNcovZ,maxNCV);
        end
        if (fHand~=1)
            fclose(fHand);
        end
    end
    
    if (visualize==true)
        [xStart1,xStart2,xEnd1,xEnd2] = Registration.Overlap.CalculateROIs(deltasZ_XY(1),imageROI1Org_XY(1),imageROI2Org_XY(1),size(im1,2),size(im2,2));
        [yStart1,yStart2,yEnd1,yEnd2] = Registration.Overlap.CalculateROIs(deltasZ_XY(2),imageROI1Org_XY(2),imageROI2Org_XY(2),size(im1,1),size(im2,1));
        
        set(Rect1,'Position',[max(xStart1,1),max(yStart1,1),max(xEnd1-xStart1,1),max(yEnd1-yStart1,1)]);
        set(Rect2,'Position',[max(xStart2,1),max(yStart2,1),max(xEnd2-xStart2,1),max(yEnd2-yStart2,1)]);
        imY1 = im1MaxChan(yStart1:yEnd1,xStart1:xEnd1);
        imY2 = im2MaxChan(yStart2:yEnd2,xStart2:xEnd2);
        
        Registration.Iterative.UpdateXYviewer(imY1,imY2,maxNcovZ,deltasZ_XY(1),deltasZ_XY(2),deltasZ_XY(3));
    end
end

%% fixup results
if (maxNcovZ-maxNCV < -0.1)
    warning('ROI normalized covariance is worse in 3D (%f) than in 2D (%f)',maxNcovZ,maxNCV);
%     maxNcovZ = max(maxNcovZ,maxNCV);
end

[xStart1,xStart2,xEnd1,xEnd2] = Registration.Overlap.CalculateROIs(deltasZ_XY(1),imageROI1Org_XY(1),imageROI2Org_XY(1),size(im1,2),size(im2,2));
[yStart1,yStart2,yEnd1,yEnd2] = Registration.Overlap.CalculateROIs(deltasZ_XY(2),imageROI1Org_XY(2),imageROI2Org_XY(2),size(im1,1),size(im2,1));
[zStart1,zStart2,zEnd1,zEnd2] = Registration.Overlap.CalculateROIs(deltasZ_XY(3),1,1,size(im1,3),size(im2,3));

overlapSize = max(xEnd1-xStart1,1) * max(yEnd1-yStart1,1) * max(zEnd1-zStart1,1);

% if (maxNcovZ>0.0 && overlapSize >= minOverlap^3)
    ultimateDeltaX = deltasZ_XY(1);
    ultimateDeltaY = deltasZ_XY(2);
    ultimateDeltaZ = deltasZ_XY(3);
    
    maxNCV = maxNcovZ;
% else
%     maxNCV = -inf;
%     ultimateDeltaX = 0;
%     ultimateDeltaY = 0;
%     ultimateDeltaZ = 0;
% end

clear imROI1 imROI2
end
