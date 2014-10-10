function [ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNcor,overlapSize] = registerTwoImages(im1,imageDataset1,im2,imageDataset2,chan,minOverlap,drawDecisionSurf,visualize)
global maxSearchSize

if (~exist('minOverlap','var') || isempty(minOverlap))
    minOverlap = 50;
end
if (isempty(maxSearchSize))
    maxSearchSize = 100;
end

if (~exist('drawDecisionSurf','var') || isempty(drawDecisionSurf))
    drawDecisionSurf = 0;
end

if (~exist('visualize','var') || isempty(visualize))
    visualize = 0;
end

fprintf('Registering %s with %s...',imageDataset1.DatasetName,imageDataset2.DatasetName);

[image1ROI,image2ROI] = calculateOverlap(imageDataset1,imageDataset2);

if (visualize==1)
    error('This is not working with a parfor, change and comment out this line');
%     delete(gcp);
%     parpool(1);
%     fig = figure;
%     
%     subplot(5,2,[1,3])
%     imagesc(max(im1(:,:,:,chan),[],3)), colormap gray, axis image
%     hold on
%     rectangle('Position',[image1ROI(1),image1ROI(2),image1ROI(3)-image1ROI(1),image1ROI(4)-image1ROI(2)],'EdgeColor','g');
%     
%     subplot(5,2,[2,4])
%     imagesc(max(im2(:,:,:,chan),[],3)), colormap gray, axis image
%     hold on
%     rectangle('Position',[image2ROI(1),image2ROI(2),image2ROI(3)-image2ROI(1),image2ROI(4)-image2ROI(2)],'EdgeColor','g');
%     
%     subplot(5,2,5)
%     imagesc(max(im1(image1ROI(2):image1ROI(4),image1ROI(1):image1ROI(3),:,chan),[],3)), colormap gray, axis image
%     hold on
%     rect1 = rectangle('Position',[1,1,image1ROI(3)-image1ROI(1),image1ROI(4)-image1ROI(2)],'EdgeColor','r');
%     
%     subplot(5,2,6)
%     imagesc(max(im2(image2ROI(2):image2ROI(4),image2ROI(1):image2ROI(3),:,chan),[],3)), colormap gray, axis image
%     hold on
%     rect2 = rectangle('Position',[1,1,image2ROI(3)-image2ROI(1),image2ROI(4)-image2ROI(2)],'EdgeColor','r');
end

minXROI1 = image1ROI(1);
maxXROI1 = image1ROI(3);
minYROI1 = image1ROI(2);
maxYROI1 = image1ROI(4);
minXROI2 = image2ROI(1);
maxXROI2 = image2ROI(3);
minYROI2 = image2ROI(2);
maxYROI2 = image2ROI(4);

maxIterX = min(maxSearchSize,min(maxXROI1-minXROI1,maxXROI2-minXROI2));
maxIterY = min(maxSearchSize,min(maxYROI1-minYROI1,maxYROI2-minYROI2));

if (maxIterX<minOverlap*3 && maxIterY<minOverlap*3 || maxIterX<10 || maxIterY<10)
    ultimateDeltaX = 0;
    ultimateDeltaY = 0;
    ultimateDeltaZ = 0;
    maxNcor = -inf;
    overlapSize = 0;
    fprintf('Does not meet minimums\n');
    return
else
    fprintf('\n');
end

imMax1 = max(im1(minYROI1:maxYROI1,minXROI1:maxXROI1,:,chan),[],3);
imMax2 = max(im2(minYROI2:maxYROI2,minXROI2:maxXROI2,:,chan),[],3);

normCovar = zeros(maxIterY*2,maxIterX*2);

% maxCovar = -inf;

totalTm = tic;
parfor deltaX = 1:maxIterX*2
%     warning('Is not running as parfor, change line and comment out this message');
    curDeltaX = deltaX-maxIterX;
    xStart1 = max(1, 1+curDeltaX);
    xStart2 = max(1, 1-curDeltaX);
    xEnd1 = xStart1 + min(size(imMax1,2)-xStart1,size(imMax2,2)-xStart2);
    xEnd2 = xStart2 + min(size(imMax2,2)-xStart2,size(imMax1,2)-xStart1);
    
    if (xEnd1-xStart1~=xEnd2-xStart2),error('Sizes dont`t match %d : %d!',xEnd1-xStart1,xEnd2-xStart2), end
    if (xEnd1-xStart1<minOverlap), continue, end
    
    im1X = imMax1(:,xStart1:xEnd1);
    im2X = imMax2(:,xStart2:xEnd2);
    
    normCoLine = zeros(maxIterY*2,1);
    
    for deltaY = 1:maxIterY*2
        curDeltaY = deltaY-maxIterY;
        yStart1 = max(1, 1+curDeltaY);
        yStart2 = max(1, 1-curDeltaY);
        yEnd1 = yStart1 + min(size(imMax1,1)-yStart1,size(imMax2,1)-yStart2);
        yEnd2 = yStart2 + min(size(imMax2,1)-yStart2,size(imMax1,1)-yStart1);
        
        if (yEnd1-yStart1~=yEnd2-yStart2),error('Sizes dont`t match %d : %d!',yEnd1-yStart1,yEnd2-yStart2), end
        if (yEnd1-yStart1<minOverlap), continue, end
        
        im1Y = im1X(yStart1:yEnd1,:);
        im2Y = im2X(yStart2:yEnd2,:);
        
        normCoLine(deltaY) = CudaMex('NormalizedCovariance',im1Y,im2Y);
        
        
%         if (visualize==1)
%             maxCovar = updateXYviewer(rect1,rect2,xStart1,yStart1,xEnd1,yEnd1,xStart2,yStart2,xEnd2,yEnd2,im1Y,im2Y,maxCovar,normCoLine,deltaX,deltaY,maxIterX,maxIterY);
%         end
    end
    
    normCovar(:,deltaX) = normCoLine;
end

tm = toc(totalTm);
hr = floor(tm/3600);
tmNew = tm - hr*3600;
mn = floor(tmNew/60);
tmNew = tmNew - mn*60;
sc = tmNew;
fprintf('Total time: %d:%02d:%04.2f\n\t average per step %5.3f\n\t average per scan line %5.3f\n\n',hr,mn,sc,tm/(maxIterX*2*maxIterY*2),tm/(maxIterX*2));

[maxNcor,I] = max(normCovar(:));
[r,c] = ind2sub(size(normCovar),I);
bestDeltaX = c-maxIterX;
bestDeltaY = r-maxIterY;

xStart1 = max(1, minXROI1+bestDeltaX);
xStart2 = max(1, minXROI1-bestDeltaX);
xEnd1 = xStart1 + min(size(im1,2)-xStart1, size(im2,2)-xStart2);
xEnd2 = xStart2 + min(size(im2,2)-xStart2, size(im1,2)-xStart1);
yStart1 = max(1, minYROI1+bestDeltaY);
yStart2 = max(1, minYROI2-bestDeltaY);
yEnd1 = yStart1 + min(size(im1,1)-yStart1, size(im2,1)-yStart2);
yEnd2 = yStart2 + min(size(im2,1)-yStart2, size(im1,1)-yStart1);

if (xEnd1-xStart1~=xEnd2-xStart2),error('Sizes dont`t match %d : %d!',xEnd1-xStart1,xEnd2-xStart2), end
if (yEnd1-yStart1~=yEnd2-yStart2),error('Sizes dont`t match %d : %d!',yEnd1-yStart1,yEnd2-yStart2), end

if (drawDecisionSurf==1)
    co = CudaMex('NormalizedCovariance',max(im1(yStart1:yEnd1,xStart1:xEnd1,:,chan),[],3),max(im2(yStart2:yEnd2,xStart2:xEnd2,:,chan),[],3));
    
    figure
    surf(normCovar,'EdgeColor','none');
    hold on
    text(c,r,maxNcor,sprintf('  \\Delta (%d,%d):%.3f',bestDeltaX,bestDeltaY,co),'Color','r','BackgroundColor','k','VerticalAlignment','bottom');
    scatter3(c,r,maxNcor,'fill');
    drawnow
end

imROI1 = squeeze(im1(yStart1:yEnd1,xStart1:xEnd1,:,chan));
imROI2 = squeeze(im2(yStart2:yEnd2,xStart2:xEnd2,:,chan));

% if (visualize==1)
%     figure(fig)
%     maxCovar = -inf;
%     subplot(5,2,5)
%     imagesc(max(imROI1,[],3)), colormap gray, axis image
%     hold on
%     rect1 = rectangle('Position',[1,1,size(imROI1,2),size(imROI1,2)],'EdgeColor','r');
%     
%     subplot(5,2,6)
%     imagesc(max(imROI2,[],3)), colormap gray, axis image
%     hold on
%     rect2 = rectangle('Position',[1,1,size(imROI2,2),size(imROI2,2)],'EdgeColor','r');
% end

maxIterZ = floor(min(imageDataset1.ZDimension,imageDataset2.ZDimension)/2);
maxIterX = 5;
maxIterY = 5;
normCovarZ = zeros(maxIterY*2,maxIterX*2,maxIterZ*2);
totalTm = tic;

%     warning('Is not running as parfor, change line and comment out this message');
parfor deltaZ = 1:maxIterZ*2
    curDeltaZ = deltaZ-maxIterZ;
    zStart1 = max(1, 1+curDeltaZ);
    zStart2 = max(1, 1-curDeltaZ);
    zEnd1 = zStart1 + min(size(imROI1,3)-zStart1,size(imROI1,3)-zStart2);
    zEnd2 = zStart2 + min(size(imROI1,3)-zStart2,size(imROI1,3)-zStart1);
    
    if (zEnd1-zStart1~=zEnd2-zStart2),error('Sizes dont`t match %d : %d!',zEnd1-zStart1,zEnd2-zStart2), end
    
    im1Z = imROI1(:,:,zStart1:zEnd1);
    im2Z = imROI2(:,:,zStart2:zEnd2);
    
    normCoSquare = zeros(maxIterY*2,maxIterX*2);
    for deltaX = 1:maxIterX*2
        curDeltaX = deltaX-maxIterX;
        xStart1 = max(1, 1+curDeltaX);
        xStart2 = max(1, 1-curDeltaX);
        xEnd1 = xStart1 + min(size(im1Z,2)-xStart1,size(im1Z,2)-xStart2);
        xEnd2 = xStart2 + min(size(im2Z,2)-xStart2,size(im2Z,2)-xStart1);
        
        if (xEnd1-xStart1~=xEnd2-xStart2),error('Sizes dont`t match %d : %d!',xEnd1-xStart1,xEnd2-xStart2), end
        if (xEnd1-xStart1<minOverlap), continue, end
        
        im1X = im1Z(:,xStart1:xEnd1,:);
        im2X = im2Z(:,xStart2:xEnd2,:);
        
        normCoLine = zeros(maxIterY*2,1);
        
        for deltaY = 1:maxIterY*2
            curDeltaY = deltaY-maxIterY;
            yStart1 = max(1, 1+curDeltaY);
            yStart2 = max(1, 1-curDeltaY);
            yEnd1 = yStart1 + min(size(im1X,1)-yStart1,size(im1X,1)-yStart2);
            yEnd2 = yStart2 + min(size(im2X,1)-yStart2,size(im2X,1)-yStart1);
            
            if (yEnd1-yStart1~=yEnd2-yStart2),error('Sizes dont`t match %d : %d!',yEnd1-yStart1,yEnd2-yStart2), end
            if (yEnd1-yStart1<minOverlap), continue, end
            
            im1Y = im1X(yStart1:yEnd1,:,:);
            im2Y = im2X(yStart2:yEnd2,:,:);
            
            normCoLine(deltaY) = CudaMex('NormalizedCovariance',im1Y,im2Y);
            
%             if (visualize==1)
%                 maxCovar = updateXYviewer(rect1,rect2,xStart1,yStart1,xEnd1,yEnd1,xStart2,yStart2,xEnd2,yEnd2,im1Y,maxCovar,normCoLine,deltaX,deltaY,maxIterX,maxIterY);
%             end
        end
        
        normCoSquare(:,deltaX) = normCoLine;
    end
    
    normCovarZ(:,:,deltaZ) = normCoSquare;
end

tm = toc(totalTm);
hr = floor(tm/3600);
tmNew = tm - hr*3600;
mn = floor(tmNew/60);
tmNew = tmNew - mn*60;
sc = tmNew;
fprintf('Total time: %d:%02d:%04.2f\n\t average per step %5.3f\n\t average per scan line %5.3f\n\t average per scan box %5.3f\n\n',...
    hr,mn,sc,tm/(maxIterZ*2*maxIterX*2*maxIterY*2),tm/(maxIterZ*2*maxIterX*2),tm/(maxIterZ*2));

[maxNcor,I] = max(normCovarZ(:));
[r,c,z] = ind2sub(size(normCovarZ),I);

ultimateDeltaX = bestDeltaX +maxIterX -c;
ultimateDeltaY = bestDeltaY +maxIterY -r;
ultimateDeltaZ = maxIterZ -z;

xStart1 = max(1, minXROI1+ultimateDeltaX);
xStart2 = max(1, minXROI1-ultimateDeltaX);
xEnd1 = xStart1 + min(size(im1,2)-xStart1, size(im2,2)-xStart2);
xEnd2 = xStart2 + min(size(im2,2)-xStart2, size(im1,2)-xStart1);
yStart1 = max(1, minYROI1+ultimateDeltaY);
yStart2 = max(1, minYROI2-ultimateDeltaY);
yEnd1 = yStart1 + min(size(im1,1)-yStart1, size(im2,1)-yStart2);
yEnd2 = yStart2 + min(size(im2,1)-yStart2, size(im1,1)-yStart1);
zStart1 = max(1, 1+ultimateDeltaZ);
zStart2 = max(1, 1-ultimateDeltaZ);
zEnd1 = zStart1 + min(size(im1,3)-zStart1, size(im2,3)-zStart2);
zEnd2 = zStart2 + min(size(im2,3)-zStart2, size(im1,3)-zStart1);

if (xEnd1-xStart1~=xEnd2-xStart2),error('Sizes dont`t match %d : %d!',xEnd1-xStart1,xEnd2-xStart2), end
if (yEnd1-yStart1~=yEnd2-yStart2),error('Sizes dont`t match %d : %d!',yEnd1-yStart1,yEnd2-yStart2), end
if (zEnd1-zStart1~=zEnd2-zStart2),error('Sizes dont`t match %d : %d!',zEnd1-zStart1,zEnd2-zStart2), end

overlapSize = (xEnd1-xStart1) * (yEnd1-yStart1) * (zEnd1-zStart1);

if (drawDecisionSurf==1)    
    co = CudaMex('NormalizedCovariance',im1(yStart1:yEnd1,xStart1:xEnd1,zStart1:zEnd1,chan),im2(yStart2:yEnd2,xStart2:xEnd2,zStart2:zEnd2,chan));
    
    figure
    surf(normCovarZ(:,:,z),'EdgeColor','none');
    hold on
    text(c,r,maxNcor,sprintf('  \\Delta (%d,%d,%d):%.3f',ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,co),'Color','r','BackgroundColor','k','VerticalAlignment','bottom');
    scatter3(c,r,maxNcor,'fill');
    drawnow
end
end

% function maxCovar = updateXYviewer(rect1,rect2,xStart1,yStart1,xEnd1,yEnd1,xStart2,yStart2,xEnd2,yEnd2,im1Y,im2Y,maxCovar,normCoLine,deltaX,deltaY,maxIterX,maxIterY)
% set(rect1,'Position',[xStart1,yStart1,xEnd1-xStart1,yEnd1-yStart1]);
% set(rect2,'Position',[xStart2,yStart2,xEnd2-xStart2,yEnd2-yStart2]);
% subplot(5,2,7)
% imagesc(max(im1Y,[],3)),colormap gray, axis image
% subplot(5,2,8)
% imagesc(max(im2Y,[],3)),colormap gray, axis image
% if (maxCovar<normCoLine(deltaY))
%     maxCovar = normCoLine(deltaY);
%     subplot(5,2,9)
%     imagesc(max(im1Y,[],3)),colormap gray, axis image
%     subplot(5,2,10)
%     imagesc(max(im2Y,[],3)),colormap gray, axis image
%     title(sprintf('(%d,%d):%1.3f',deltaX-maxIterX,deltaY-maxIterY,maxCovar));
% end
% drawnow
% end