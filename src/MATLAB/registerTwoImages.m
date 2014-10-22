function [ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNCV,overlapSize] = registerTwoImages(im1,imageDataset1,im2,imageDataset2,chan,minOverlap,maxSearchSize,showDecisionSurf,visualize,device)
clear global Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar MaxCovar SubImBest1 SubImBest2 DecisionFig DecisionAxes
global Rect1 Rect2

%% check inputs
if (~exist('minOverlap','var') || isempty(minOverlap))
    minOverlap = 25;
end
if (~exist('maxSearchSize','var') || isempty(maxSearchSize))
    maxSearchSize = 100;
end

if (~exist('showDecisionSurf','var') || isempty(showDecisionSurf))
    showDecisionSurf = 0;
end

if (~exist('visualize','var') || isempty(visualize))
    visualize = 0;
end

if (~exist('device','var') || isempty(device))
    device = 1;
end

%% setup and early out

fprintf('%s --> %s\n',imageDataset1.DatasetName,imageDataset2.DatasetName);

[imageROI1,imageROI2,minXdist,minYdist] = calculateOverlap(imageDataset1,imageDataset2);

maxIterX = maxSearchSize;
maxIterY = maxSearchSize;

if (minXdist>maxSearchSize-minOverlap || minYdist>maxSearchSize-minOverlap)
    ultimateDeltaX = 0;
    ultimateDeltaY = 0;
    ultimateDeltaZ = 0;
    maxNCV = -inf;
    overlapSize = 0;
    fprintf('%s --> %s Does not meet minimums\n',imageDataset1.DatasetName,imageDataset2.DatasetName);
    return
end

imMax1 = max(im1(:,:,:,chan),[],3);
imMax2 = max(im2(:,:,:,chan),[],3);

%% run 2-D case
totalTm = tic;

if (visualize==1)
    setupVisualizer(imMax1,imMax2,imageROI1,imageROI2,imageDataset1,imageDataset2);
end

normCovar = iterateOverX(maxIterX,maxIterY,imMax1,imMax2,imageROI1(1),imageROI2(1),imageROI1(2),imageROI2(2),0,minOverlap,visualize,device);

tm = toc(totalTm);

%% find the best answer and check results
[maxNCV,I] = max(normCovar(:));
[r,c] = ind2sub(size(normCovar),I);
bestDeltaX = c-maxIterX;
bestDeltaY = r-maxIterY;

fprintf('%s, per step %5.3f, per scan line %5.3f, NVC:%04.3f at (%d,%d)\n',...
    printTime(tm),tm/(maxIterX*2*maxIterY*2),tm/(maxIterX*2),maxNCV,bestDeltaX,bestDeltaY);

[xStart1,xStart2,xEnd1,xEnd2] = calculateROIs(bestDeltaX,imageROI1(1),imageROI2(1),size(im1,2),size(im2,2));
[yStart1,yStart2,yEnd1,yEnd2] = calculateROIs(bestDeltaY,imageROI1(2),imageROI2(2),size(im1,1),size(im2,1));

if (visualize==1)
    set(Rect1,'Position',[xStart1,yStart1,xEnd1-xStart1,yEnd1-yStart1]);
    set(Rect2,'Position',[xStart2,yStart2,xEnd2-xStart2,yEnd2-yStart2]);
end

curCovar = CudaMex('NormalizedCovariance',...
    max(im1(yStart1:yEnd1,xStart1:xEnd1,:,chan),[],3),...
    max(im2(yStart2:yEnd2,xStart2:xEnd2,:,chan),[],3));

if (maxNCV ~= curCovar)
    warning('ROI normalized covariance (%f) did not match the max (%f)',curCovar,maxNCV);
end

if (showDecisionSurf)% || maxNCV~=curCovar)
    drawDecisionSurf(normCovar,c,r,bestDeltaX,bestDeltaY,0,maxNCV,curCovar,1,imageDataset1,imageDataset2);
end

%% run 3-D case
newROI1 = [xStart1,yStart1,imageROI1(3),xEnd1,yEnd1,imageROI1(6)];
newROI2 = [xStart2,yStart2,imageROI2(3),xEnd2,yEnd2,imageROI2(6)];

imC1 = squeeze(im1(:,:,:,chan));
imC2 = squeeze(im2(:,:,:,chan));

maxIterZ = floor(min(newROI1(6)-newROI1(3),newROI2(6)-newROI2(3))/2);
maxIterX = 5;
maxIterY = 5;
totalTm = tic;

if (visualize==1)
    setupVisualizer(imMax1,imMax2,newROI1,newROI2,imageDataset1,imageDataset2);
end

normCovarZ = iterateOverZ(maxIterZ,maxIterX,maxIterY,imC1,imC2,newROI1(1),newROI2(1),newROI1(2),newROI2(2),newROI1(3),...
    newROI2(3),minOverlap,visualize,device);

tm = toc(totalTm);

%% find the best answer and check
[maxNcovZ,I] = max(normCovarZ(:));
[r,c,z] = ind2sub(size(normCovarZ),I);

ultimateDeltaX = bestDeltaX + c-maxIterX;
ultimateDeltaY = bestDeltaY + r-maxIterY;
ultimateDeltaZ = z - maxIterZ;

fprintf('%s, per step %5.3f, per scan line %5.3f, per scan box %5.3f, NVC:%04.3f at (%d,%d,%d)\n',...
    printTime(tm),tm/(maxIterZ*2*maxIterX*2*maxIterY*2),tm/(maxIterZ*2*maxIterX*2),tm/(maxIterZ*2),maxNcovZ,ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ);

if (c-maxIterX~=0 || r-maxIterY~=0)
    fprintf('A better delta was found when looking in Z. Change in deltas=(%d,%d,%d) Old NCV:%f new:%f\n', c-maxIterX,r-maxIterY,ultimateDeltaZ,maxNcovZ,maxNCV);
end

[xStart1,xStart2,xEnd1,xEnd2] = calculateROIs(ultimateDeltaX,imageROI1(1),imageROI2(1),size(im1,2),size(im2,2));
[yStart1,yStart2,yEnd1,yEnd2] = calculateROIs(ultimateDeltaY,imageROI1(2),imageROI2(2),size(im1,1),size(im2,1));
[zStart1,zStart2,zEnd1,zEnd2] = calculateROIs(ultimateDeltaZ,1,1,size(im1,3),size(im2,3));

curCovar = CudaMex('NormalizedCovariance',...
    im1(yStart1:yEnd1,xStart1:xEnd1,zStart1:zEnd1,chan),...
    im2(yStart2:yEnd2,xStart2:xEnd2,zStart2:zEnd2,chan));

if (maxNcovZ ~= curCovar)
    warning('ROI normalized covariance (%f) did not match the max (%f)',curCovar,maxNCV);
    maxNcovZ = max(maxNcovZ,curCovar);
end

if (showDecisionSurf)% || c-maxIterX~=0 || r-maxIterY~=0 || maxNcovZ~=curCovar)
    drawDecisionSurf(normCovarZ(:,:,z),c,r,ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNcovZ,curCovar,2,imageDataset1,imageDataset2);
end

if (visualize==1)
    set(Rect1,'Position',[xStart1,yStart1,xEnd1-xStart1,yEnd1-yStart1]);
    set(Rect2,'Position',[xStart2,yStart2,xEnd2-xStart2,yEnd2-yStart2]);
    clear global normCovar normCovarZ Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar MaxCovar SubImBest1 SubImBest2 DecisionFig DecisionAxes
end

overlapSize = (xEnd1-xStart1) * (yEnd1-yStart1) * (zEnd1-zStart1);

maxNCV = maxNcovZ;

clear imROI1 imROI2
end

function normCoCube = iterateOverZ(maxIterZ,maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,zStart1,zStart2,minOverlap,visualize,device)
normCoCube = zeros(maxIterY*2,maxIterX*2,maxIterZ*2);
imZ1 = [];
imZ2 = [];
for delta = 1:maxIterZ*2
    curDelta = delta-maxIterZ;
    [start1,start2,end1,end2] = calculateROIs(curDelta,zStart1,zStart2,size(im1,3),size(im2,3));
    if (end1-start1<minOverlap || end2-start2<minOverlap), continue, end
    
    imZ1 = im1(:,:,start1:end1);
    imZ2 = im2(:,:,start2:end2);
    normCoCube(:,:,delta) = iterateOverX(maxIterX,maxIterY,imZ1,imZ2,xStart1,xStart2,...
        yStart1,yStart2,curDelta,minOverlap,visualize,device);
end
clear('imZ1');
clear('imZ2');
end

function normCoSquare = iterateOverX(maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,curDeltaZ,minOverlap,visualize,device)
global Rect1 Rect2
normCoSquare = zeros(maxIterY*2,maxIterX*2);
imX1 = [];
imX2 = [];
for delta = 1:maxIterX*2
    curDelta = delta-maxIterX;
    [start1,start2,end1,end2] = calculateROIs(curDelta,xStart1,xStart2,size(im1,2),size(im2,2));
    if (end1-start1<minOverlap || end2-start2<minOverlap), continue, end
    
    if (visualize==1)
        pos1 = get(Rect1,'Position');
        pos2 = get(Rect2,'Position');
        set(Rect1,'Position',[start1,pos1(2),end1-start1,pos1(4)]);
        set(Rect2,'Position',[start2,pos2(2),end2-start2,pos2(4)]);
    end
    
    imX1 = im1(:,start1:end1,:);
    imX2 = im2(:,start2:end2,:);
    normCoSquare(:,delta) = iterateOverY(maxIterY,imX1,imX2,curDelta,yStart1,yStart2,curDeltaZ,minOverlap,visualize,device);
end
clear('imX1');
clear('imX2');
end

function normCoLine = iterateOverY(maxIterY,im1,im2,curDeltaX,yStart1,yStart2,curDeltaZ,minOverlap,visualize,device)
global Rect1 Rect2
normCoLine = zeros(maxIterY*2,1);
imY1 = [];
imY2 = [];
for delta = 1:maxIterY*2
    curDelta = delta-maxIterY;
    [start1,start2,end1,end2] = calculateROIs(curDelta,yStart1,yStart2,size(im1,1),size(im2,1));
    if (end1-start1<minOverlap || end2-start2<minOverlap), continue, end
    
    imY1 = im1(start1:end1,:,:);
    imY2 = im2(start2:end2,:,:);
    
    if (device~=0)
        normCoLine(delta) = CudaMex('NormalizedCovariance',imY1,imY2,device);
    else
        imY1 = double(imY1);
        imY2 = double(imY2);
        sig1 = sqrt(var(imY1(:)));
        sig2 = sqrt(var(imY2(:)));
        
        mean1 = mean(imY1(:));
        mean2 = mean(imY2(:));
        
        imSub1 = imY1 - mean1;
        imSub2 = imY2 - mean2;
        
        imMul = imSub1.*imSub2;
        
        numerator = sum(imMul(:));
        
        normCoLine(delta) = numerator / (numel(imY1)*sig1*sig2);
    end
    
    if (visualize==1)
        pos1 = get(Rect1,'Position');
        pos2 = get(Rect2,'Position');
        set(Rect1,'Position',[pos1(1),start1,pos1(3),end1-start1]);
        set(Rect2,'Position',[pos2(1),start2,pos2(3),end2-start2]);
        updateXYviewer(imY1,imY2,normCoLine(delta),curDeltaX,curDelta,curDeltaZ);
    end
    
    if (normCoLine(delta)>1 || normCoLine(delta)<-1)
        warning('Recived a NCV out of bounds:%f, overlap:(%d,%d,%d)',normCoLine(delta),size(imY1,2),size(imY1,1),size(imY1,3));
        normCoLine(delta) = 0;
    end
end
clear('imY1');
clear('imY2');
end

function [start1,start2,end1,end2] = calculateROIs(delta,oldStart1,oldStart2,size1,size2)
if (oldStart1==1 && oldStart2~=1)
    start1 = 1;
else
    start1 = max(1, oldStart1+delta);
end

if (oldStart2==1 && oldStart1~=1)
    start2 = 1;
else
    start2 = max(1, oldStart2-delta);
end

minSize = min(size1-start1,size2-start2);
end1 = start1 + minSize;
end2 = start2 + minSize;

if (end1-start1~=end2-start2),error('Sizes dont`t match %d : %d!',end1-start1,end2-start2), end
end

function setupVisualizer(im1,im2,image1ROI,image2ROI,imageData1,imageData2)
global Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar

MaxCovar = -inf;
Fig = figure;

SubImOrg1 = subplot(2,2,1);
imagesc(im1,'Parent',SubImOrg1);
colormap(SubImOrg1,'gray');
axis(SubImOrg1,'image');
hold(SubImOrg1);
rectangle('Position',[image1ROI(1),image1ROI(2),image1ROI(4)-image1ROI(1),image1ROI(5)-image1ROI(2)],'EdgeColor','r','Parent',SubImOrg1);
Rect1 = rectangle('Position',[image1ROI(1),image1ROI(2),image1ROI(4)-image1ROI(1),image1ROI(5)-image1ROI(2)],'EdgeColor','g','Parent',SubImOrg1);
title(SubImOrg1,imageData1.DatasetName,'Interpreter','none');

SubImOrg2 = subplot(2,2,2);
imagesc(im2,'Parent',SubImOrg2);
colormap(SubImOrg2,'gray');
axis(SubImOrg2,'image');
hold(SubImOrg2);
rectangle('Position',[image2ROI(1),image2ROI(2),image2ROI(4)-image2ROI(1),image2ROI(5)-image2ROI(2)],'EdgeColor','r','Parent',SubImOrg2);
Rect2 = rectangle('Position',[image2ROI(1),image2ROI(2),image2ROI(4)-image2ROI(1),image2ROI(5)-image2ROI(2)],'EdgeColor','g','Parent',SubImOrg2);
title(SubImOrg2,imageData2.DatasetName,'Interpreter','none');

SubImBest1 = subplot(2,2,3);
SubImBest2 = subplot(2,2,4);
end

function updateXYviewer(im1,im2,normCovar,curDeltaX,curDeltaY,curDeltaZ)
global MaxCovar SubImBest1 SubImBest2

if (MaxCovar<normCovar)
    MaxCovar = normCovar;
    imagesc(max(im1,[],3),'Parent',SubImBest1)
    colormap(SubImBest1,'gray')
    axis(SubImBest1,'image')
    imagesc(max(im2,[],3),'Parent',SubImBest2)
    colormap(SubImBest2,'gray')
    axis(SubImBest2,'image')
    titleText = sprintf('Best Deltas (%d,%d,%d):%1.3f',curDeltaX,curDeltaY,curDeltaZ,normCovar);
    title(SubImBest1,titleText);
    title(SubImBest2,titleText);
end

if (mod(curDeltaY,5)==0)
    drawnow
end
end

function drawDecisionSurf(decisionArray,c,r,deltaX,deltaY,deltaZ,covariance1,covariance2,subPlotIdx,imageDataset1,imageDataset2)
global DecisionFig DecisionAxes
if (isempty(DecisionFig))
    DecisionFig = figure;
    subplot1 = subplot(1,2,1);
    title(subplot1,imageDataset1.DatasetName,'Interpreter','none');
    subplot2 = subplot(1,2,2);
    title(subplot2,imageDataset2.DatasetName,'Interpreter','none');
    DecisionAxes = [subplot1, subplot2];
end

surf(decisionArray,'EdgeColor','none','Parent',DecisionAxes(subPlotIdx));
hold(DecisionAxes(subPlotIdx))
text(c,r,covariance1,sprintf('  \\Delta (%d,%d,%d):%.3f',deltaX,deltaY,deltaZ,covariance2),...
    'Color','r','BackgroundColor','k','VerticalAlignment','bottom','Parent',DecisionAxes(subPlotIdx));

scatter3(c,r,covariance1,'fill','Parent',DecisionAxes(subPlotIdx));
drawnow
hold(DecisionAxes(subPlotIdx),'off');
end
