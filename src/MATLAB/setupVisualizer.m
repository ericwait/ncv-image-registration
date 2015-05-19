function setupVisualizer(im1,im2,image1ROI,image2ROI,imageData1,imageData2)
global Fig Rect1 Rect2 SubImOrg1 SubImOrg2 SubImBest1 SubImBest2 MaxCovar

MaxCovar = -inf;
Fig = figure;

SubImOrg1 = subplot(2,2,1);
imagesc(im1,'Parent',SubImOrg1);
colormap(SubImOrg1,'gray');
colorbar
axis(SubImOrg1,'image');
hold(SubImOrg1,'on');
rectPos = [max(image1ROI(1),1),max(image1ROI(2),1),max(image1ROI(4)-image1ROI(1),1),max(image1ROI(5)-image1ROI(2),1)];
rectangle('Position',rectPos,'EdgeColor','r','Parent',SubImOrg1);
Rect1 = rectangle('Position',rectPos,'EdgeColor','g','Parent',SubImOrg1);
title(SubImOrg1,imageData1.DatasetName,'Interpreter','none');

SubImOrg2 = subplot(2,2,2);
imagesc(im2,'Parent',SubImOrg2);
colormap(SubImOrg2,'gray');
colorbar
axis(SubImOrg2,'image');
hold(SubImOrg2,'on');
rectPos = [max(image2ROI(1),1),max(image2ROI(2),1),max(image2ROI(4)-image2ROI(1),1),max(image2ROI(5)-image2ROI(2),1)];
rectangle('Position',rectPos,'EdgeColor','r','Parent',SubImOrg2);
Rect2 = rectangle('Position',rectPos,'EdgeColor','g','Parent',SubImOrg2);
title(SubImOrg2,imageData2.DatasetName,'Interpreter','none');

SubImBest1 = subplot(2,2,3);
colorbar
SubImBest2 = subplot(2,2,4);
colorbar
end
