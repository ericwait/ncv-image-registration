function DrawDecisionSurf(decisionArray,x,y,c,deltaX,deltaY,deltaZ,covariance1,covariance2,subPlotIdx,imageDataset1,imageDataset2,maxIterX,maxIterY)
global DecisionFig DecisionAxes
if (isempty(DecisionFig))
    DecisionFig = figure;
    subplot1 = subplot(1,2,1);
    title(subplot1,imageDataset1.DatasetName,'Interpreter','none');
    subplot2 = subplot(1,2,2);
    title(subplot2,imageDataset2.DatasetName,'Interpreter','none');
    DecisionAxes = [subplot1, subplot2];
end

if (~exist('maxIterX','var') || isempty(maxIterX))
    maxIterX = 0;
end

if (~exist('maxIterY','var') || isempty(maxIterY))
    maxIterY = 0;
end

[X,Y] = meshgrid(1:size(decisionArray,2),1:size(decisionArray,1));
X = X - maxIterX;
Y = Y - maxIterY;
x = x - maxIterX;
y = y - maxIterY;
surf(X,Y,decisionArray,'EdgeColor','none','Parent',DecisionAxes(subPlotIdx));
hold(DecisionAxes(subPlotIdx),'on')
text(x,y,covariance1,sprintf('  \\Delta (%d,%d,%d,%d):%.3f',deltaX,deltaY,deltaZ,c,covariance2),...
    'Color','r','BackgroundColor','k','VerticalAlignment','bottom','Parent',DecisionAxes(subPlotIdx));

scatter3(x,y,covariance1,'fill','Parent',DecisionAxes(subPlotIdx));

drawnow

xlabel(DecisionAxes(subPlotIdx),'Delta X')
ylabel(DecisionAxes(subPlotIdx),'Delta Y')
zlabel(DecisionAxes(subPlotIdx),'Normalized Covariance')

xlim(DecisionAxes(subPlotIdx),[min(X(:)),max(X(:))])
ylim(DecisionAxes(subPlotIdx),[min(Y(:)),max(Y(:))])
end
