function DrawBoxesLines(figHandle,axisHandle,imageDatasets,chan,curDataset)
hold(axisHandle,'on');

minXPos = min([imageDatasets(:).xMinPos]);
minYPos = min([imageDatasets(:).yMinPos]);

lw = 2;
grn = [0 0.4 0];
gry = [0 0 0];

for i=1:length(imageDatasets)
    rectangle('Position',...
        [imageDatasets(i).xMinPos-minXPos-imageDatasets(i).xDelta,...
         imageDatasets(i).yMinPos-minYPos-imageDatasets(i).yDelta,...
         imageDatasets(i).XDimension, imageDatasets(i).YDimension],...
         'EdgeColor',grn,'LineStyle','-.','LineWidth',lw,'Parent',axisHandle);
    rectangle('Position',...
        [imageDatasets(i).xMinPos-minXPos,...
         imageDatasets(i).yMinPos-minYPos,...
         imageDatasets(i).XDimension, imageDatasets(i).YDimension],...
         'EdgeColor','r','LineStyle',':','LineWidth',lw,'Parent',axisHandle);
    
    centerCur = [imageDatasets(i).xMinPos-minXPos + imageDatasets(i).XDimension/2,...
        (imageDatasets(i).yMinPos-minYPos) + imageDatasets(i).YDimension/2];
    
    centerParent = [imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos+ imageDatasets(imageDatasets(i).ParentDelta).XDimension/2,...
        imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos + imageDatasets(imageDatasets(i).ParentDelta).YDimension/2];
    
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b','LineWidth',lw,'Parent',axisHandle);
end

for i=1:length(imageDatasets)
    centerCur = [imageDatasets(i).xMinPos-minXPos + imageDatasets(i).XDimension/2,...
        imageDatasets(i).yMinPos-minYPos + imageDatasets(i).YDimension/2];
    centerParent = [imageDatasets(imageDatasets(i).ParentDelta).xMinPos-minXPos + imageDatasets(imageDatasets(i).ParentDelta).XDimension/2,...
        imageDatasets(imageDatasets(i).ParentDelta).yMinPos-minYPos + imageDatasets(imageDatasets(i).ParentDelta).YDimension/2];
    
    posNum = regexp(imageDatasets(i).DatasetName,[regexptranslate('escape',curDataset) '(.*)'],'tokens','once');
    if (isempty(posNum))
        posNum = num2str(i);
    end
    text(centerCur(1)+20,centerCur(2)-20,posNum,'color','r','FontSize',14,'BackgroundColor',gry,'Parent',axisHandle,'Interpreter','none');
    text((centerParent(1)+centerCur(1))/2,(centerParent(2)+centerCur(2))/2,...
        sprintf('(%d,%d,%d):%04.2f',imageDatasets(i).xDelta,imageDatasets(i).yDelta,imageDatasets(i).zDelta,imageDatasets(i).NCV),...
        'color',[0.749,0.749,0],'FontSize',10,'BackgroundColor',gry,'Parent',axisHandle,'Interpreter','none');
end

title(axisHandle,sprintf('Cannel:%d',chan),'Interpreter','none','Color','w');

set(figHandle,'Color',[0 0 0]);
set(axisHandle,'Color',[0 0 0]);
set(figHandle,'PaperPositionMode','auto');
set(figHandle,'InvertHardcopy','off');
set(figHandle,'Units','normalized','Position',[0 0 1 1]);
set(axisHandle,'Box', 'off','units','normalized','Position',[0 0 1 1]);
axis(axisHandle,'image')
hold(axisHandle,'off');
end
