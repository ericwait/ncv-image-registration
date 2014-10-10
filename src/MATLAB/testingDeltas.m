function testingDeltas(outImage, outImageColor,imageDatasets)
figure
hold off
[img, colorIdx] = max(outImage(:,:,:),[],3);
[r,c] = ndgrid(1:size(outImage,1),1:size(outImage,2));
idx = sub2ind(size(outImage),r,c,colorIdx);
pixelColors = outImageColor(idx);
cmap = hsv(12);
cmap = cmap(randi(12, max(double(pixelColors(:)))+1),:);
alpha = cat(3, reshape(cmap(pixelColors+1,1),size(pixelColors)), reshape(cmap(pixelColors+1,2),size(pixelColors)), reshape(cmap(pixelColors+1,3),size(pixelColors)));
imgC = alpha .* repmat(mat2gray(img),[1 1 3]);

imagesc(img), colormap gray
hold on

minXPos = min([imageDatasets(:).xMinPos]);
minYPos = min([imageDatasets(:).yMinPos]);
minZPos = min([imageDatasets(:).zMinPos]);
maxXPos = max([imageDatasets(:).xMaxPos]);
maxYPos = max([imageDatasets(:).yMaxPos]);
maxZPos = max([imageDatasets(:).zMaxPos]);
minXPixelPhysicalSize = min([imageDatasets(:).XPixelPhysicalSize]);
minYPixelPhysicalSize = min([imageDatasets(:).YPixelPhysicalSize]);
minZPixelPhysicalSize = min([imageDatasets(:).ZPixelPhysicalSize]);
imageWidth = round((maxXPos-minXPos)/minXPixelPhysicalSize +1);
imageHeight = round((maxYPos-minYPos)/minYPixelPhysicalSize +1);
imageDepth = round((maxZPos-minZPos)/minZPixelPhysicalSize +1);

lw = 2;
grn = [0 0.4 0];
gry = [0 0 0];

for i=1:length(imageDatasets)
    rectangle('Position',...
        [(imageDatasets(i).yMinPos-minYPos-imageDatasets(i).yDelta)/imageDatasets(i).YPixelPhysicalSize...
        (imageDatasets(i).xMinPos-minXPos-imageDatasets(i).xDelta)/imageDatasets(i).XPixelPhysicalSize...
        imageDatasets(i).YDimension imageDatasets(i).XDimension],'EdgeColor',grn,'LineStyle','-.','LineWidth',lw);
    rectangle('Position',...
        [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize...
        imageDatasets(i).YDimension imageDatasets(i).XDimension],'EdgeColor','r','LineStyle',':','LineWidth',lw);
    
    centerCur = [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize + imageDatasets(i).YDimension/2 ...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize + imageDatasets(i).XDimension/2];
    
    centerParent = [(imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos)/imageDatasets(imageDatasets(i).ParentDelta).YPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).YDimension/2 ...
        (imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos)/imageDatasets(imageDatasets(i).ParentDelta).XPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).XDimension/2];
    
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b','LineWidth',lw);
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b','LineWidth',lw);
end

for i=1:length(imageDatasets)
    centerCur = [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize + imageDatasets(i).YDimension/2 ...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize + imageDatasets(i).XDimension/2];
    
    text(centerCur(1)+20,centerCur(2)-20,num2str(i),'color','r','FontSize',14,'BackgroundColor',gry);
end

set(gcf,'Color',[0 0 0]);
set(gca,'Color',[0 0 0]);
set(gcf,'PaperPositionMode','auto');
set(gcf,'InvertHardcopy','off');
set(gcf,'Position',[20 20 2550 1434]);
axis image
hold off

figure
image(imgC);
hold on
for i=1:length(imageDatasets)
    rectangle('Position',...
        [(imageDatasets(i).yMinPos-minYPos-imageDatasets(i).yDelta)/imageDatasets(i).YPixelPhysicalSize...
        (imageDatasets(i).xMinPos-minXPos-imageDatasets(i).xDelta)/imageDatasets(i).XPixelPhysicalSize...
        imageDatasets(i).YDimension imageDatasets(i).XDimension],'EdgeColor',grn,'LineStyle','-.','LineWidth',lw);
    rectangle('Position',...
        [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize...
        imageDatasets(i).YDimension imageDatasets(i).XDimension],'EdgeColor','r','LineStyle',':','LineWidth',lw);
    
    centerCur = [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize + imageDatasets(i).YDimension/2 ...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize + imageDatasets(i).XDimension/2];
    
    centerParent = [(imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos)/imageDatasets(imageDatasets(i).ParentDelta).YPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).YDimension/2 ...
        (imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos)/imageDatasets(imageDatasets(i).ParentDelta).XPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).XDimension/2];
    
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b','LineWidth',lw);
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b','LineWidth',lw);
end

for i=1:length(imageDatasets)
    centerCur = [(imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize + imageDatasets(i).YDimension/2 ...
        (imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize + imageDatasets(i).XDimension/2];
    
    text(centerCur(1)+20,centerCur(2)-20,num2str(i),'color','r','FontSize',14,'BackgroundColor',gry);
end

set(gcf,'Color',[0 0 0]);
set(gca,'Color',[0 0 0]);
set(gcf,'PaperPositionMode','auto');
set(gcf,'InvertHardcopy','off');
axis ij
set(gcf,'Position',[120 120 2550 1434]);
axis image
hold off

drawnow

end