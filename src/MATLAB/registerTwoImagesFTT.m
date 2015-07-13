function [ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNCV,overlapSize] = registerTwoImagesFTT(im1,imageDataset1,im2,...
    imageDataset2,minOverlap,maxSearchSize,logFile)

%% check inputs
if (~exist('minOverlap','var') || isempty(minOverlap))
    minOverlap = 25;
end
if (~exist('maxSearchSize','var') || isempty(maxSearchSize))
    maxSearchSize = 100;
end
if (~exist('logFile','var') || isempty(logFile))
    logFile = 1;
end

%% setup and early out

if (logFile~=1)
    fHand = fopen(logFile,'at');
else
    fHand = 1;
end
fprintf(fHand,'%s \n\t--> %s\n',imageDataset1.DatasetName,imageDataset2.DatasetName);
if (fHand~=1)
    fclose(fHand);
end

[imageROI1,imageROI2,~,~,padding] = calculateOverlap(imageDataset1,imageDataset2,[maxSearchSize,maxSearchSize,0]);

maxNCV = -inf;
bestChan = 0;

im1ROI = im1(imageROI1(2):imageROI1(5),imageROI1(1):imageROI1(4),imageROI1(3):imageROI1(6),:,:);
im2ROI = im2(imageROI2(2):imageROI2(5),imageROI2(1):imageROI2(4),imageROI2(3):imageROI2(6),:,:);

%% run 2-D case
totalTm = tic;

im1MaxROI = squeeze(max(im1ROI,[],3));
im2MaxROI = squeeze(max(im2ROI,[],3));
for c=1:imageDataset1.NumberOfChannels
    [deltas,curNCV] = Helper.getMaxNCVdeltas(im1MaxROI(:,:,c),im2MaxROI(:,:,c),minOverlap^2,maxSearchSize);
    if (curNCV>maxNCV)
        bestChan = c;
        maxNCV = curNCV;
        bestDeltas = [Helper.flipFirstTwoDims(deltas),0] - padding;
    end
end

tm = toc(totalTm);

bestDeltas(3) = 0;

if (logFile~=1)
    fHand = fopen(logFile,'at');
else
    fHand = 1;
end
fprintf(fHand,'\t%s, NVC:%04.3f at (%d,%d) on channel:%d\n',...
    printTime(tm),maxNCV,bestDeltas(1),bestDeltas(2),c);
if (fHand~=1)
    fclose(fHand);
end

%% run 3-D case
if (size(im1,3)>1)
    totalTm = tic;
    
    [deltasZ,maxNcovZ] = Helper.getMaxNCVdeltas(im1ROI(:,:,:,bestChan),im2ROI(:,:,:,bestChan),minOverlap^3,maxSearchSize);
    deltasZ = Helper.flipFirstTwoDims(deltasZ) - padding;
    
    tm = toc(totalTm);
    
    if (logFile~=1)
        fHand = fopen(logFile,'at');
    else
        fHand = 1;
    end
    fprintf(fHand,'\t%s, NVC:%04.3f at (%d,%d,%d)\n',...
        printTime(tm),maxNcovZ,deltasZ(1),deltasZ(2),deltasZ(3));
    
    changeDelta = bestDeltas - deltasZ;
    if (changeDelta(1)~=0 || changeDelta(2)~=0)
        fprintf(fHand,'\tA better delta was found when looking in Z. Change in deltas=(%d,%d,%d) Old NCV:%f new:%f\n', changeDelta(1),changeDelta(2),changeDelta(3),maxNcovZ,maxNCV);
    end
    if (fHand~=1)
        fclose(fHand);
    end
end
%% fixup results

if (abs(maxNcovZ-maxNCV)>0.00001)
    warning('ROI normalized covariance (%f) did not match the max (%f)',maxNCV,maxNcovZ);
    maxNcovZ = max(maxNcovZ,maxNCV);
end

[xStart1,xStart2,xEnd1,xEnd2] = calculateROIs(deltasZ(1)-padding(1),imageROI1(1),imageROI2(1),size(im1,2),size(im2,2));
[yStart1,yStart2,yEnd1,yEnd2] = calculateROIs(deltasZ(2)-padding(2),imageROI1(2),imageROI2(2),size(im1,1),size(im2,1));
[zStart1,zStart2,zEnd1,zEnd2] = calculateROIs(deltasZ(3),1,1,size(im1,3),size(im2,3));

overlapSize = (xEnd1-xStart1) * (yEnd1-yStart1) * (zEnd1-zStart1);
ultimateDeltaX = deltasZ(1);
ultimateDeltaY = deltasZ(2);
ultimateDeltaZ = deltasZ(3);

maxNCV = maxNcovZ;

if (overlapSize < minOverlap^3)
    maxNCV = -inf;
end

clear imROI1 imROI2
end
