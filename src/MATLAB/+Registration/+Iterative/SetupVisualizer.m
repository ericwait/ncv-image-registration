function SetupVisualizer(im1,im2,image1ROI,image2ROI,imageData1,imageData2)
global Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar

MaxCovar = -inf;
Fig = figure;

g = colormap('gray');
gB = brighten(g,0.3);

set(Fig,'Color',[.2 .2 .2]);

SubImOrg1 = subplot(2,2,1);
imagesc(im1,'Parent',SubImOrg1);
colormap(SubImOrg1,gB);
%colorbar
axis(SubImOrg1,'image');
hold(SubImOrg1,'on');
rectPos = [max(image1ROI(1),1),max(image1ROI(2),1),max(image1ROI(4)-image1ROI(1)+1,1),max(image1ROI(5)-image1ROI(2)+1,1)];
rectangle('Position',rectPos,'EdgeColor','r','Parent',SubImOrg1,'linewidth',1.5);
Rect1 = rectangle('Position',rectPos,'EdgeColor','g','Parent',SubImOrg1,'linewidth',1.5);
title(SubImOrg1,imageData1.DatasetName,'Interpreter','none','Color',[.75,.75,.75],'FontSize',24);
set(SubImOrg1,'xtick',[])
set(SubImOrg1,'xticklabel',[])
set(SubImOrg1,'ytick',[])
set(SubImOrg1,'yticklabel',[])

SubImOrg2 = subplot(2,2,2);
imagesc(im2,'Parent',SubImOrg2);
colormap(SubImOrg2,gB);
brighten(.2);
%colorbar
axis(SubImOrg2,'image');
hold(SubImOrg2,'on');
rectPos = [max(image2ROI(1),1),max(image2ROI(2),1),max(image2ROI(4)-image2ROI(1)+1,1),max(image2ROI(5)-image2ROI(2)+1,1)];
rectangle('Position',rectPos,'EdgeColor','r','Parent',SubImOrg2,'linewidth',1.5);
Rect2 = rectangle('Position',rectPos,'EdgeColor','g','Parent',SubImOrg2,'linewidth',1.5);
title(SubImOrg2,imageData2.DatasetName,'Interpreter','none','Color',[.75,.75,.75],'FontSize',24);
set(SubImOrg2,'xtick',[])
set(SubImOrg2,'xticklabel',[])
set(SubImOrg2,'ytick',[])
set(SubImOrg2,'yticklabel',[])

SubImBest1 = subplot(2,2,3);
set(SubImBest1,'xtick',[])
set(SubImBest1,'xticklabel',[])
set(SubImBest1,'ytick',[])
set(SubImBest1,'yticklabel',[])
%colorbar
SubImBest2 = subplot(2,2,4);
set(SubImBest2,'xtick',[])
set(SubImBest2,'xticklabel',[])
set(SubImBest2,'ytick',[])
set(SubImBest2,'yticklabel',[])
%colorbar

startCorner = 100;
set(Fig,'unit','pixel','Position',[startCorner,startCorner,1920+200,1080+113])
end
