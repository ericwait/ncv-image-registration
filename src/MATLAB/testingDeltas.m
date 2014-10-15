function testingDeltas(outImage, outImageColor,imageDatasets,chan)

%% setup colormap for the colored image
[img, colorIdx] = max(outImage(:,:,:),[],3);
[r,c] = ndgrid(1:size(outImage,1),1:size(outImage,2));
idx = sub2ind(size(outImage),r,c,colorIdx);
pixelColors = outImageColor(idx);
cmap = hsv(12);
cmap = cmap(randi(12, max(double(pixelColors(:)))+1),:);
alpha = cat(3, reshape(cmap(pixelColors+1,1),size(pixelColors)), reshape(cmap(pixelColors+1,2),size(pixelColors)), reshape(cmap(pixelColors+1,3),size(pixelColors)));
imgC = alpha .* repmat(mat2gray(img),[1 1 3]);

fig = figure;
imagesc(img);
ax = gca;
colormap(ax,'gray');

drawBoxesLines(fig,ax,imageDatasets,chan);

fig = figure;
image(imgC);
ax = gca;
drawBoxesLines(fig,ax,imageDatasets,chan);

drawnow

end

function drawBoxesLines(figHandle,axisHandle,imageDatasets,chan)
hold(axisHandle);

minXPos = min([imageDatasets(:).xMinPos]);
minYPos = min([imageDatasets(:).yMinPos]);

lw = 2;
grn = [0 0.4 0];
gry = [0 0 0];

for i=1:length(imageDatasets)
    rectangle('Position',...
        [(imageDatasets(i).xMinPos-minXPos-imageDatasets(i).xDelta)/imageDatasets(i).XPixelPhysicalSize,...
        (imageDatasets(i).yMinPos-minYPos-imageDatasets(i).yDelta)/imageDatasets(i).YPixelPhysicalSize,...
        imageDatasets(i).XDimension, imageDatasets(i).YDimension],'EdgeColor',grn,'LineStyle','-.','LineWidth',lw,'Parent',axisHandle);
    rectangle('Position',...
        [(imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize...
        (imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize...
        imageDatasets(i).XDimension, imageDatasets(i).YDimension],'EdgeColor','r','LineStyle',':','LineWidth',lw,'Parent',axisHandle);
    
    centerCur = [(imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize + imageDatasets(i).XDimension/2,...
        (imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize + imageDatasets(i).YDimension/2];
    
    centerParent = [(imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos)/imageDatasets(imageDatasets(i).ParentDelta).XPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).XDimension/2,...
        (imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos)/imageDatasets(imageDatasets(i).ParentDelta).YPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).YDimension/2];
    
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b','LineWidth',lw,'Parent',axisHandle);
end

for i=1:length(imageDatasets)
    centerCur = [(imageDatasets(i).xMinPos-minXPos)/imageDatasets(i).XPixelPhysicalSize + imageDatasets(i).XDimension/2,...
        (imageDatasets(i).yMinPos-minYPos)/imageDatasets(i).YPixelPhysicalSize + imageDatasets(i).YDimension/2];
    centerParent = [(imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos)/imageDatasets(imageDatasets(i).ParentDelta).XPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).XDimension/2,...
        (imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos)/imageDatasets(imageDatasets(i).ParentDelta).YPixelPhysicalSize + imageDatasets(imageDatasets(i).ParentDelta).YDimension/2];
    
    text(centerCur(1)+20,centerCur(2)-20,num2str(i),'color','r','FontSize',14,'BackgroundColor',gry,'Parent',axisHandle);
    text((centerParent(1)+centerCur(1))/2,(centerParent(2)+centerCur(2))/2,...
        sprintf('(%d,%d,%d):%04.2f',imageDatasets(i).xDelta,imageDatasets(i).yDelta,imageDatasets(i).zDelta,imageDatasets(i).NCV),...
        'color',[0.749,0.749,0],'FontSize',10,'BackgroundColor',gry,'Parent',axisHandle);
end

title(axisHandle,sprintf('Cannel:%d',chan),'Interpreter','none','Color','w');

set(figHandle,'Color',[0 0 0]);
set(axisHandle,'Color',[0 0 0]);
set(figHandle,'PaperPositionMode','auto');
set(figHandle,'InvertHardcopy','off');
set(figHandle,'Position',[20 20 2550 1434]);
axis(axisHandle,'image')
hold(axisHandle,'off');
end