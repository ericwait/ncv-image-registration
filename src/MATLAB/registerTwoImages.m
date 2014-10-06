function [ultimateDeltaX,ultimateDeltaY,ultimateDeltaZ,maxNcor,overlapSize] = registerTwoImages(im1,imageDataset1,im2,imageDataset2,chan,drawDecisionSurf,visualize)
global minOverlap

if (isempty(minOverlap))
    minOverlap = 50;
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

minX1 = image1ROI(1);
maxX1 = image1ROI(3);
minY1 = image1ROI(2);
maxY1 = image1ROI(4);
minX2 = image2ROI(1);
maxX2 = image2ROI(3);
minY2 = image2ROI(2);
maxY2 = image2ROI(4);

maxIterX = min(150,min(maxX1-minX1,maxX2-minX2));
maxIterY = min(150,min(maxY1-minY1,maxY2-minY2));

if (maxIterX<minOverlap && maxIterY<minOverlap)
    ultimateDeltaX = 0;
    ultimateDeltaY = 0;
    ultimateDeltaZ = 0;
    maxNcor = inf;
    fprintf('Does not meet minimums\n');
    return
else
    fprintf('\n');
end

imMax1 = max(im1(image1ROI(2):image1ROI(4),image1ROI(1):image1ROI(3),:,chan),[],3);
imMax2 = max(im2(image2ROI(2):image2ROI(4),image2ROI(1):image2ROI(3),:,chan),[],3);

normCovar = zeros(maxIterY*2,maxIterX*2);

% maxCovar = -inf;

totalTm = tic;
parfor deltaX = 1:maxIterX*2
%     warning('Is not running as parfor, change line and comment out this message');
    xStart1 = max(maxIterX-deltaX, 1);
    xStart2 = max(deltaX-maxIterX, 1);
    xEnd1 = min(size(imMax1,2)-xStart2+xStart1,size(imMax1,2));
    xEnd2 = min(size(imMax2,2)-xStart1+xStart2,size(imMax2,2));
    
    if (xEnd1-xStart1~=xEnd2-xStart2),error('Sizes dont`t match %d : %d!',xEnd1-xStart1,xEnd2-xStart2), end
    
    im1X = imMax1(:,xStart1:xEnd1);
    im2X = imMax2(:,xStart2:xEnd2);
    
    normCoLine = zeros(maxIterY*2,1);
    
    for deltaY = 1:maxIterY*2
        yStart1 = max(maxIterY-deltaY, 1);
        yStart2 = max(deltaY-maxIterY, 1);
        yEnd1 = min(size(im1X,1)-yStart2+yStart1,size(im1X,1));
        yEnd2 = min(size(im2X,1)-yStart1+yStart2,size(im2X,1));
        
        if (yEnd1-yStart1~=yEnd2-yStart2),error('Sizes dont`t match %d : %d!',yEnd1-yStart1,yEnd2-yStart2), end
        
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
bestDeltaX = c-maxIterX+1;
bestDeltaY = r-maxIterY+1;

xStart1 = max(minX1-bestDeltaX, 1);
xStart2 = max(minX2+bestDeltaX, 1);
xEnd1 = min(maxX1-bestDeltaX,size(im1,2));
xEnd2 = min(maxX2+bestDeltaX,size(im2,2));
yStart1 = max(minY1-bestDeltaY, 1);
yStart2 = max(minY2+bestDeltaY, 1);
yEnd1 = min(maxY1-bestDeltaY,size(im1,1));
yEnd2 = min(maxY2+bestDeltaY,size(im2,1));

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
maxIterX = 10;
maxIterY = 10;
normCovarZ = zeros(maxIterY*2,maxIterX*2,maxIterZ*2);
totalTm = tic;

parfor deltaZ = 1:maxIterZ*2
%     warning('Is not running as parfor, change line and comment out this message');
    zStart1 = max(maxIterZ-deltaZ, 1);
    zStart2 = max(deltaZ-maxIterZ, 1);
    zEnd1 = min(size(imROI1,3)-zStart2+zStart1,size(imROI1,3));
    zEnd2 = min(size(imROI2,3)-zStart1+zStart2,size(imROI2,3));
    
    if (zEnd1-zStart1~=zEnd2-zStart2),error('Sizes dont`t match %d : %d!',zEnd1-zStart1,zEnd2-zStart2), end
    
    im1Z = imROI1(:,:,zStart1:zEnd1);
    im2Z = imROI2(:,:,zStart2:zEnd2);
    
    normCoSquare = zeros(maxIterY*2,maxIterX*2);
    for deltaX = 1:maxIterX*2
        xStart1 = max(maxIterX-deltaX, 1);
        xStart2 = max(deltaX-maxIterX, 1);
        xEnd1 = min(size(imROI1,2)-xStart2+xStart1,size(imROI1,2));
        xEnd2 = min(size(imROI2,2)-xStart1+xStart2,size(imROI2,2));
        
        if (xEnd1-xStart1~=xEnd2-xStart2),error('Sizes dont`t match %d : %d!',xEnd1-xStart1,xEnd2-xStart2), end
        
        im1X = im1Z(:,xStart1:xEnd1,:);
        im2X = im2Z(:,xStart2:xEnd2,:);
        
        normCoLine = zeros(maxIterY*2,1);
        
        for deltaY = 1:maxIterY*2
            yStart1 = max(maxIterY-deltaY, 1);
            yStart2 = max(deltaY-maxIterY, 1);
            yEnd1 = min(size(im1X,1)-yStart2+yStart1,size(im1X,1));
            yEnd2 = min(size(im2X,1)-yStart1+yStart2,size(im2X,1));
            
            if (yEnd1-yStart1~=yEnd2-yStart2),error('Sizes dont`t match %d : %d!',yEnd1-yStart1,yEnd2-yStart2), end
            
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

ultimateDeltaX = bestDeltaX + c - maxIterX +1;
ultimateDeltaY = bestDeltaY + r - maxIterY +1;
ultimateDeltaZ = z - maxIterZ +1;

xStart1 = max(minX1-ultimateDeltaX, 1);
xStart2 = max(minX2+ultimateDeltaX, 1);
xEnd1 = min(maxX1-ultimateDeltaX,size(im1,2));
xEnd2 = min(maxX2+ultimateDeltaX,size(im2,2));
yStart1 = max(minY1-ultimateDeltaY, 1);
yStart2 = max(minY2+ultimateDeltaY, 1);
yEnd1 = min(maxY1-ultimateDeltaY,size(im1,1));
yEnd2 = min(maxY2+ultimateDeltaY,size(im2,1));
zStart1 = max(-ultimateDeltaZ+1, 1);
zStart2 = max(ultimateDeltaZ+1, 1);
zEnd1 = min(size(im1,3)-ultimateDeltaZ,size(im1,3));
zEnd2 = min(size(im1,3)+ultimateDeltaZ,size(im2,3));

if (xEnd1-xStart1~=xEnd2-xStart2),error('Sizes dont`t match %d : %d!',xEnd1-xStart1,xEnd2-xStart2), end
if (yEnd1-yStart1~=yEnd2-yStart2),error('Sizes dont`t match %d : %d!',yEnd1-yStart1,yEnd2-yStart2), end
if (zEnd1-zStart1~=zEnd2-zStart2),error('Sizes dont`t match %d : %d!',zEnd1-zStart1,zEnd2-zStart2), end

overlapSize = (xEnd1-xStart1) * (yEnd1-yStart1) * (zEnd1-zStart1);

if (drawDecisionSurf==1)    
    co = CudaMex('NormalizedCovariance',max(im1(yStart1:yEnd1,xStart1:xEnd1,zStart1:zEnd1,chan),[],3),max(im2(yStart2:yEnd2,xStart2:xEnd2,zStart2:zEnd2,chan),[],3));
    
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