function [ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNCV,overlapSize] = registerTwoImages(im1,imageDataset1,im2,imageDataset2,chan,minOverlap,maxSearchSize,showDecisionSurf,visualize)
clear global Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar MaxCovar SubImBest1 SubImBest2 DecisionFig DecisionAxes
global Rect1 Rect2

%% check inputs
if (~exist('minOverlap','var') || isempty(minOverlap))
    minOverlap = 50;
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

%% setup and early out

fprintf('Registering %s\n\t-->%s...',imageDataset1.DatasetName,imageDataset2.DatasetName);

[imageROI1,imageROI2] = calculateOverlap(imageDataset1,imageDataset2);

maxIterX = min(maxSearchSize,min(imageROI1(4)-imageROI1(1),imageROI2(4)-imageROI2(1)));
maxIterY = min(maxSearchSize,min(imageROI1(5)-imageROI1(2),imageROI2(5)-imageROI2(2)));

if (maxIterX<minOverlap && maxIterY<minOverlap)
    ultimateDeltaX = 0;
    ultimateDeltaY = 0;
    ultimateDeltaZ = 0;
    maxNCV = -inf;
    overlapSize = 0;
    fprintf('Does not meet minimums\n');
    return
else
    fprintf('\n');
end

imMax1 = max(im1(:,:,:,chan),[],3);
imMax2 = max(im2(:,:,:,chan),[],3);

%% run 2-D case
totalTm = tic;

if (visualize==1)
    setupVisualizer(imMax1,imMax2,imageROI1,imageROI2,imageDataset1,imageDataset2);
    normCovar = iterateOverXDraw(maxIterX,maxIterY,imMax1,imMax2,imageROI1(1),imageROI2(1),imageROI1(2),imageROI2(2),0);
else
    normCovar = iterateOverX(maxIterX,maxIterY,imMax1,imMax2,imageROI1(1),imageROI2(1),imageROI1(2),imageROI2(2));
end

tm = toc(totalTm);
hr = floor(tm/3600);
tmNew = tm - hr*3600;
mn = floor(tmNew/60);
tmNew = tmNew - mn*60;
sc = tmNew;
fprintf('Total time: %d:%02d:%04.2f\n\t average per step %5.3f\n\t average per scan line %5.3f\n\n',...
    hr,mn,sc,tm/(maxIterX*2*maxIterY*2),tm/(maxIterX*2));

%% find the best answer and check results
[maxNCV,I] = max(normCovar(:));
[r,c] = ind2sub(size(normCovar),I);
bestDeltaX = c-maxIterX;
bestDeltaY = r-maxIterY;

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

if (showDecisionSurf || maxNCV~=curCovar)
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
    normCovarZ = iterateOverZDraw(maxIterZ,maxIterX,maxIterY,imC1,imC2,newROI1(1),newROI2(1),newROI1(2),newROI2(2),newROI1(3),newROI2(3));
else
    normCovarZ = iterateOverZ(maxIterZ,maxIterX,maxIterY,imC1,imC2,newROI1(1),newROI2(1),newROI1(2),newROI2(2),newROI1(3),newROI2(3));
end

tm = toc(totalTm);
hr = floor(tm/3600);
tmNew = tm - hr*3600;
mn = floor(tmNew/60);
tmNew = tmNew - mn*60;
sc = tmNew;
fprintf('Total time: %d:%02d:%04.2f\n\t average per step %5.3f\n\t average per scan line %5.3f\n\t average per scan box %5.3f\n\n',...
    hr,mn,sc,tm/(maxIterZ*2*maxIterX*2*maxIterY*2),tm/(maxIterZ*2*maxIterX*2),tm/(maxIterZ*2));

%% find the best answer and check
[maxNcovZ,I] = max(normCovarZ(:));
[r,c,z] = ind2sub(size(normCovarZ),I);

ultimateDeltaX = bestDeltaX + c-maxIterX;
ultimateDeltaY = bestDeltaY + r-maxIterY;
ultimateDeltaZ = z - maxIterZ;

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

if (showDecisionSurf || c-maxIterX~=0 || r-maxIterY~=0 || maxNcovZ~=curCovar)
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

function normCoCube = iterateOverZ(maxIterZ,maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,zStart1,zStart2)
normCoCube = zeros(maxIterY*2,maxIterX*2,maxIterZ*2);

parfor delta = 1:maxIterZ*2
    curDelta = delta-maxIterZ;
    [start1,start2,end1,end2] = calculateROIs(curDelta,zStart1,zStart2,size(im1,3),size(im2,3));
    
    normCoCube(:,:,delta) = iterateOverX(maxIterX,maxIterY,im1(:,:,start1:end1),im2(:,:,start2:end2),xStart1,xStart2,yStart1,yStart2);
end
end

function normCoCube = iterateOverZDraw(maxIterZ,maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,zStart1,zStart2)
normCoCube = zeros(maxIterY*2,maxIterX*2,maxIterZ*2);

for delta = 1:maxIterZ*2
    curDelta = delta-maxIterZ;
    [start1,start2,end1,end2] = calculateROIs(curDelta,zStart1,zStart2,size(im1,3),size(im2,3));
    
    normCoCube(:,:,delta) = iterateOverXDraw(maxIterX,maxIterY,im1(:,:,start1:end1),im2(:,:,start2:end2),xStart1,xStart2,yStart1,yStart2,curDelta);
end
end

function normCoSquare = iterateOverX(maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2)
normCoSquare = zeros(maxIterY*2,maxIterX*2);
parfor delta = 1:maxIterX*2
    curDelta = delta-maxIterX;
    [start1,start2,end1,end2] = calculateROIs(curDelta,xStart1,xStart2,size(im1,2),size(im2,2));
    
    normCoSquare(:,delta) = iterateOverY(maxIterY,im1(:,start1:end1,:),im2(:,start2:end2,:),yStart1,yStart2);
end
end

function normCoSquare = iterateOverXDraw(maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,curDeltaZ)
global Rect1 Rect2
normCoSquare = zeros(maxIterY*2,maxIterX*2);
for delta = 1:maxIterX*2
    curDelta = delta-maxIterX;
    [start1,start2,end1,end2] = calculateROIs(curDelta,xStart1,xStart2,size(im1,2),size(im2,2));
    pos1 = get(Rect1,'Position');
    pos2 = get(Rect2,'Position');
    set(Rect1,'Position',[start1,pos1(2),end1-start1,pos1(4)]);
    set(Rect2,'Position',[start2,pos2(2),end2-start2,pos2(4)]);
    
    normCoSquare(:,delta) = iterateOverYDraw(maxIterY,im1(:,start1:end1,:),im2(:,start2:end2,:),curDelta,yStart1,yStart2,curDeltaZ);
end
end

function normCoLine = iterateOverY(maxIterY,im1,im2,yStart1,yStart2)
normCoLine = zeros(maxIterY*2,1);

parfor delta = 1:maxIterY*2
    curDelta = delta-maxIterY;
    [start1,start2,end1,end2] = calculateROIs(curDelta,yStart1,yStart2,size(im1,1),size(im2,1));
    
    normCoLine(delta) = CudaMex('NormalizedCovariance',im1(start1:end1,:,:),im2(start2:end2,:,:));
end
end

function normCoLine = iterateOverYDraw(maxIterY,im1,im2,curDeltaX,yStart1,yStart2,curDeltaZ)
global Rect1 Rect2
normCoLine = zeros(maxIterY*2,1);

for delta = 1:maxIterY*2
    curDelta = delta-maxIterY;
    [start1,start2,end1,end2] = calculateROIs(curDelta,yStart1,yStart2,size(im1,1),size(im2,1));
    
    normCoLine(delta) = CudaMex('NormalizedCovariance',im1(start1:end1,:,:),im2(start2:end2,:,:));
    pos1 = get(Rect1,'Position');
    pos2 = get(Rect2,'Position');
    set(Rect1,'Position',[pos1(1),start1,pos1(3),end1-start1]);
    set(Rect2,'Position',[pos2(1),start2,pos2(3),end2-start2]);
    updateXYviewer(im1(start1:end1,:,:),im2(start2:end2,:,:),normCoLine(delta),curDeltaX,curDelta,curDeltaZ);
end
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
hold off
end