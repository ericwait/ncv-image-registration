factorLowRes = 5;
factorHighRes = 3;

imLowPath = ['D:\Users\Eric\Documents\Programming\LEVer3d\src\MATLAB\data\DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568\images\movie\singles\bigX',...
    num2str(factorLowRes),'.bmp'];

imLow = imread(imLowPath);
figure, imshow(imLow);

imLowPlacePath = ['D:\Users\Eric\Documents\Programming\LEVer3d\src\MATLAB\data\DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568\images\movie\x',...
    num2str(factorLowRes), '\close_t_000_r_000.bmp'];

imLowPlace = imread(imLowPlacePath);
figure, imshow(imLowPlace);

imHighPlacePath = ['D:\Users\Eric\Documents\Programming\LEVer3d\src\MATLAB\data\DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568\images\movie\x',...
    num2str(factorHighRes), '\close_t_000_r_000.bmp'];

imHighPlace = imread(imHighPlacePath);
figure, imshow(imHighPlace);

imLowOrgPath = ['D:\Users\Eric\Documents\Programming\Images\22mo2 wmSVZ Unmixed\DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568_Montage_wDelta\movie\x',...
    num2str(factorLowRes), '\DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568_c1_t0001_z0001.tif'];

imLowSizes = size(imread(imLowOrgPath));

imOrgPath = 'D:\Users\Eric\Documents\Programming\Images\22mo2 wmSVZ Unmixed\DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568_Montage_wDelta\DAPI Olig2-514 GFAP-488 Dcx-647 Laminin-Cy3 Bcatenin-568_c1_t0001_z0001.tif';
imOrgSizes = size(imread(imOrgPath));

%% set all of these
roiXstart = 30;
roiYstart = 325;
roiXend = 3211;
roiYend = 1533;

lowPlaceStartX = 34;
lowPlaceStartY = 188;
lowPlaceEndX = 1883;
lowPlaceEndY = 891;

highPlaceStartX = 99;
highPlaceStartY = 56;
highPlaceEndX = 1818;
highPlaceEndY = 1022;

centerXorg = 8000;
centerYorg = 2000;

%% org stuff
% imOrg = tiffReader('uint8',4);

%% auto run from here
imLowRoi = imLow(roiYstart:roiYend,roiXstart:roiXend,:);

xRadius = 1023;
yRadius = 575;

lowXstartIdx = centerXorg-xRadius*factorLowRes;
lowXendIdx = centerXorg+xRadius*factorLowRes;
lowYstartIdx = centerYorg-yRadius*factorLowRes;
lowYendIdx = centerYorg+yRadius*factorLowRes;

if (lowXendIdx>imOrgSizes(2))
    lowXendIdx = imOrgSizes(2);
    lowXstartIdx = max(1,imOrgSizes(2)-xRadius*2*factorLowRes);
elseif (lowXstartIdx<1)
    lowStartIdx = 1;
    lowXendIdx = min(imOrgSizes(2),1+xRadius*2*factorLowRes);
end
if (lowYendIdx>imOrgSizes(1))
    lowYendIdx = imOrgSizes(1);
    lowYstartIdx = max(1,imOrgSizes(1)-yRadius*2*factorLowRes);
elseif (lowYstartIdx<1)
    lowYstartIdx = 1;
    lowYendIdx = min(imOrgSizes(1),1+yRadius*2*factorLowRes);
end

highXstartIdx = centerXorg-xRadius*factorHighRes;
highXendIdx = centerXorg+xRadius*factorHighRes;
highYstartIdx = centerYorg-yRadius*factorHighRes;
highYendIdx = centerYorg+yRadius*factorHighRes;

if (highXendIdx>imOrgSizes(2))
    highXendIdx = imOrgSizes(2);
    highXstartIdx = max(1,imOrgSizes(2)-xRadius*2*factorHighRes);
elseif (highXstartIdx<1)
    highXstartIdx = 1;
    highXendIdx = min(imOrgSizes(2),1+xRadius*2*factorHighRes);
end
if (highYendIdx>imOrgSizes(1))
    highYendIdx = imOrgSizes(1);
    highYstartIdx = max(1,imOrgSizes(1)-yRadius*2*factorHighRes);
elseif (highYstartIdx<1)
    highYstartIdx = 1;
    highYendIdx = min(imOrgSizes(1),yRadius*2*factorHighRes);
end

% figure, imagesc(max(imOrg,[],3)); colormap gray
% rectangle('Position',[lowXstartIdx,lowYstartIdx,lowXendIdx-lowXstartIdx,lowYendIdx-lowYstartIdx],'EdgeColor','w');
% rectangle('Position',[highXstartIdx,highYstartIdx,highXendIdx-highXstartIdx,highYendIdx-highYstartIdx],'EdgeColor','w');
% axis equal

vox2pX = size(imLowRoi,2)/imLowSizes(2);
vox2pY = size(imLowRoi,1)/imLowSizes(1);

boxXstart = (highXstartIdx-lowXstartIdx)/factorLowRes * vox2pX;
boxYstart = (highYstartIdx-lowYstartIdx)/factorLowRes * vox2pY;
boxXend = (highXendIdx-lowXstartIdx)/factorLowRes * vox2pX;
boxYend = (highYendIdx-lowYstartIdx)/factorLowRes * vox2pY;

xZoomStart = [lowPlaceStartX,lowPlaceEndX-lowPlaceStartX+1];
yZoomStart = [lowPlaceStartY,lowPlaceEndY-lowPlaceStartY+1];
xZoomEnd = [highPlaceStartX,highPlaceEndX-highPlaceStartX+1];
yZoomEnd = [highPlaceStartY,highPlaceEndY-highPlaceStartY+1];

xCropStart = [1,size(imLowRoi,2)];
yCropStart = [1,size(imLowRoi,1)];
xCropEnd = [boxXstart,boxXend];
yCropEnd = [boxYstart,boxYend];

imBackground = imLowPlace;
imBackground(lowPlaceStartY-2:lowPlaceEndY+4,lowPlaceStartX-2:lowPlaceEndX+4,:) = 65; 

figure
imshow(imLowRoi);
hold on
rectangle('Position',[boxXstart,boxYstart,boxXend-boxXstart,boxYend-boxYstart],'EdgeColor','w');

set(gcf,'Position',[100 100 1920 1080])
set(gca,'Position',[0 0 1 1]);

% movieName = ['x', num2str(factorLowRes), 'zoom.avi'];
% vidObj = VideoWriter(movieName,'Uncompressed AVI');
% vidObj.FrameRate = 60;
% 
% open(vidObj);

rootDir = sprintf('.\\movieFiles\\%d_to_%d\\',factorLowRes,factorHighRes);
if (~exist(rootDir,'file'))
    if (~exist('.\movieFiles','file'))
        mkdir('.\movieFiles');
    end
    mkdir(rootDir);
end

imageSeq = 1;
n=60;
for i=0:n-1
xlim(xCropStart+i/(n-1)*(xCropEnd-xCropStart));
ylim(yCropStart+i/(n-1)*(yCropEnd-yCropStart));

placeX = round(xZoomStart+i/(n-1)*(xZoomEnd-xZoomStart));
placeY = round(yZoomStart+i/(n-1)*(yZoomEnd-yZoomStart));

imageData = getframe(gca);
imLowPlace = imresize(imageData.cdata,[placeY(2) placeX(2)]);

curIm = imBackground;
curIm(placeY(1):placeY(1)+placeY(2)-1,placeX(1):placeX(1)+placeX(2)-1,:) = imLowPlace;

if (i==0)
    for j=1:14
        filename = [rootDir sprintf('%05d',imageSeq) '.tif'];
        imwrite(curIm,filename,'tif','Compression','lzw');
        imageSeq = imageSeq +1;
        %writeVideo(vidObj,im2frame(curIm));
    end
end

filename = [rootDir sprintf('%05d',imageSeq) '.tif'];
imwrite(curIm,filename,'tif','Compression','lzw');
imageSeq = imageSeq +1;
% writeVideo(vidObj,im2frame(curIm));
end

% writeVideo(vidObj,im2frame(curIm));

%close(vidObj);
