function [ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNCV,overlapSize] = registerTwoImages(im1,imageDataset1,im2,...
    imageDataset2,minOverlap,maxSearchSize,logFile,showDecisionSurf,visualize)
clear global Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar MaxCovar SubImBest1 SubImBest2 DecisionFig DecisionAxes
global Rect1 Rect2

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
if (~exist('showDecisionSurf','var') || isempty(showDecisionSurf))
    showDecisionSurf = 0;
end
if (~exist('visualize','var') || isempty(visualize))
    visualize = 0;
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

[imageROI1,imageROI2,~,~] = calculateOverlap(imageDataset1,imageDataset2);

maxIterX = maxSearchSize;
maxIterY = maxSearchSize;

if (ndims(im1)>3)
    imMax1 = squeeze(max(im1,[],3));
else
    imMax1 = im1;
end

if (ndims(im2)>3)
    imMax2 = squeeze(max(im2,[],3));
else
    imMax2 = im2;
end
normCovar = zeros(maxIterY*2,maxIterX*2,imageDataset1.NumberOfChannels);

%% run 2-D case
totalTm = tic;

if (visualize==1)
    for c=1:imageDataset1.NumberOfChannels
        setupVisualizer(imMax1(:,:,c),imMax2(:,:,c),imageROI1,imageROI2,imageDataset1,imageDataset2);
        normCovar(:,:,c) = iterateOverX(maxIterX,maxIterY,imMax1(:,:,c),imMax2(:,:,c),imageROI1(1),imageROI2(1),imageROI1(2),imageROI2(2),0,minOverlap,visualize);
    end
else
    for c=1:imageDataset1.NumberOfChannels
        normCovar(:,:,c) = iterateOverXpar(maxIterX,maxIterY,imMax1(:,:,c),imMax2(:,:,c),imageROI1(1),imageROI2(1),imageROI1(2),imageROI2(2),0,minOverlap,visualize);
    end
end

tm = toc(totalTm);


%% find the best answer and check results
[maxNCV,I] = max(normCovar(:));
[y,x,c] = ind2sub(size(normCovar),I);
bestDeltaX = x-maxIterX;
bestDeltaY = y-maxIterY;

if (logFile~=1)
    fHand = fopen(logFile,'at');
else
    fHand = 1;
end
fprintf(fHand,'\t%s, per step %5.3f, per scan line %5.3f, NVC:%04.3f at (%d,%d) on channel:%d\n',...
    printTime(tm),tm/(maxIterX*2*maxIterY*2*imageDataset1.NumberOfChannels),tm/(maxIterX*2*imageDataset1.NumberOfChannels),...
    maxNCV,bestDeltaX,bestDeltaY,c);
if (fHand~=1)
    fclose(fHand);
end

[xStart1,xStart2,xEnd1,xEnd2] = calculateROIs(bestDeltaX,imageROI1(1),imageROI2(1),size(im1,2),size(im2,2));
[yStart1,yStart2,yEnd1,yEnd2] = calculateROIs(bestDeltaY,imageROI1(2),imageROI2(2),size(im1,1),size(im2,1));

if (visualize==1)
    set(Rect1,'Position',[max(xStart1,1),max(yStart1,1),max(xEnd1-xStart1,1),max(yEnd1-yStart1,1)]);
    set(Rect2,'Position',[max(xStart2,1),max(yStart2,1),max(xEnd2-xStart2,1),max(yEnd2-yStart2,1)]);
end

curCovar = NormalizedCovariance(...
    max(im1(yStart1:yEnd1,xStart1:xEnd1,:,c),[],3),...
    max(im2(yStart2:yEnd2,xStart2:xEnd2,:,c),[],3));

if (abs(maxNCV-curCovar)>0.00001)
    warning('ROI normalized covariance (%f) did not match the max (%f)',curCovar,maxNCV);
end

if (showDecisionSurf)% || maxNCV~=curCovar)
    drawDecisionSurf(normCovar(:,:,c),x,y,c,bestDeltaX,bestDeltaY,0,maxNCV,curCovar,1,imageDataset1,imageDataset2,maxIterX,maxIterY);
end

%% run 3-D case
if (size(im1,3)>1)
    newROI1 = [xStart1,yStart1,imageROI1(3),xEnd1,yEnd1,imageROI1(6)];
    newROI2 = [xStart2,yStart2,imageROI2(3),xEnd2,yEnd2,imageROI2(6)];
    
    maxIterZ = floor(min(newROI1(6)-newROI1(3),newROI2(6)-newROI2(3))/2);
    maxIterX = 5;
    maxIterY = 5;
    totalTm = tic;
    
    if (visualize==1)
        setupVisualizer(imMax1(:,:,c),imMax2(:,:,c),newROI1,newROI2,imageDataset1,imageDataset2);
        normCovarZ = iterateOverZ(maxIterZ,maxIterX,maxIterY,im1(:,:,:,c),im2(:,:,:,c),newROI1(1),newROI2(1),newROI1(2),newROI2(2),newROI1(3),...
            newROI2(3),minOverlap,visualize);
    else
        normCovarZ = iterateOverZpar(maxIterZ,maxIterX,maxIterY,im1(:,:,:,c),im2(:,:,:,c),newROI1(1),newROI2(1),newROI1(2),newROI2(2),newROI1(3),...
            newROI2(3),minOverlap,visualize);
    end
    
    tm = toc(totalTm);
    
    %% find the best answer and check
    [maxNcovZ,I] = max(normCovarZ(:));
    [y,x,z] = ind2sub(size(normCovarZ),I);
    
    ultimateDeltaX = bestDeltaX + x-maxIterX;
    ultimateDeltaY = bestDeltaY + y-maxIterY;
    ultimateDeltaZ = z - maxIterZ;
    
    if (logFile~=1)
        fHand = fopen(logFile,'at');
    else
        fHand = 1;
    end
    fprintf(fHand,'\t%s, per step %5.3f, per scan line %5.3f, per scan box %5.3f, NVC:%04.3f at (%d,%d,%d)\n',...
        printTime(tm),tm/(maxIterZ*2*maxIterX*2*maxIterY*2),tm/(maxIterZ*2*maxIterX*2),tm/(maxIterZ*2),maxNcovZ,ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ);
    
    if (x-maxIterX~=0 || y-maxIterY~=0)
        fprintf(fHand,'\tA better delta was found when looking in Z. Change in deltas=(%d,%d,%d) Old NCV:%f new:%f\n', c-maxIterX,y-maxIterY,ultimateDeltaZ,maxNcovZ,maxNCV);
    end
    if (fHand~=1)
        fclose(fHand);
    end
else
    maxNcovZ = curCovar;
    ultimateDeltaX = bestDeltaX;
    ultimateDeltaY = bestDeltaY;
    ultimateDeltaZ = 0;
end

[xStart1,xStart2,xEnd1,xEnd2] = calculateROIs(ultimateDeltaX,imageROI1(1),imageROI2(1),size(im1,2),size(im2,2));
[yStart1,yStart2,yEnd1,yEnd2] = calculateROIs(ultimateDeltaY,imageROI1(2),imageROI2(2),size(im1,1),size(im2,1));
[zStart1,zStart2,zEnd1,zEnd2] = calculateROIs(ultimateDeltaZ,1,1,size(im1,3),size(im2,3));

curCovar = NormalizedCovariance(...
    im1(yStart1:yEnd1,xStart1:xEnd1,zStart1:zEnd1,c),...
    im2(yStart2:yEnd2,xStart2:xEnd2,zStart2:zEnd2,c));

if (abs(maxNcovZ-curCovar)>0.00001)
    warning('ROI normalized covariance (%f) did not match the max (%f)',curCovar,maxNcovZ);
    maxNcovZ = max(maxNcovZ,curCovar);
end

if (showDecisionSurf && size(im1,3)>1)% || c-maxIterX~=0 || r-maxIterY~=0 || maxNcovZ~=curCovar)
    drawDecisionSurf(normCovarZ(:,:,z),x,y,c,ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNcovZ,curCovar,2,imageDataset1,imageDataset2);
end

if (visualize==1)
    set(Rect1,'Position',[max(xStart1,1),max(yStart1,1),max(xEnd1-xStart1,1),max(yEnd1-yStart1,1)]);
    set(Rect2,'Position',[max(xStart2,1),max(yStart2,1),max(xEnd2-xStart2,1),max(yEnd2-yStart2,1)]);
    clear global normCovar normCovarZ Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar MaxCovar SubImBest1 SubImBest2 DecisionFig DecisionAxes
end

overlapSize = (xEnd1-xStart1) * (yEnd1-yStart1) * (zEnd1-zStart1);

maxNCV = maxNcovZ;

clear imROI1 imROI2
end
