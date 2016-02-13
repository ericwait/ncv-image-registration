function DrawBoxesLines(figHandle,axisHandle,imageDatasets,minPos_XY,chan,curDataset)
hold(axisHandle,'on');

lw = 2;
grn = [0 0.4 0];
gry = [0 0 0];

for i=1:length(imageDatasets)
    newStarts = imageDatasets(i).MinPos-minPos_XY-imageDatasets(i).Delta;
    oldStarts = imageDatasets(i).MinPos-minPos_XY;
    rectangle('Position',...
        [newStarts(1), newStarts(2), imageDatasets(i).Dimensions(1), imageDatasets(i).Dimensions(2)],...
         'EdgeColor',grn,'LineStyle','-.','LineWidth',lw,'Parent',axisHandle);
    rectangle('Position',...
        [oldStarts(1), oldStarts(2), imageDatasets(i).Dimensions(1), imageDatasets(i).Dimensions(2)],...
         'EdgeColor','r','LineStyle',':','LineWidth',lw,'Parent',axisHandle);
    
    centerCur = imageDatasets(i).MinPos-minPos_XY + imageDatasets(i).Dimensions/2;
    centerParent = imageDatasets(imageDatasets(i).ParentDelta).MinPos-minPos_XY+imageDatasets(imageDatasets(i).ParentDelta).Dimensions/2;
    
    plot([centerParent(1) centerCur(1)],[centerParent(2) centerCur(2)],'b','LineWidth',lw,'Parent',axisHandle);
end

for i=1:length(imageDatasets)
    centerCur = imageDatasets(i).MinPos-minPos_XY + imageDatasets(i).Dimensions/2;
    centerParent = imageDatasets(imageDatasets(i).ParentDelta).MinPos-minPos_XY+imageDatasets(imageDatasets(i).ParentDelta).Dimensions/2;
    
    posNum = regexp(imageDatasets(i).DatasetName,[regexptranslate('escape',curDataset) '(.*)'],'tokens','once');
    if (isempty(posNum))
        posNum = num2str(i);
    end
    text(centerCur(1)+20,centerCur(2)-20,posNum,'color','r','FontSize',14,'BackgroundColor',gry,'Parent',axisHandle,'Interpreter','none');
    text((centerParent(1)+centerCur(1))/2,(centerParent(2)+centerCur(2))/2,...
        sprintf('(%d,%d,%d):%04.2f',imageDatasets(i).Delta(1),imageDatasets(i).Delta(2),imageDatasets(i).Delta(3),imageDatasets(i).NCV),...
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
