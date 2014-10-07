function makeSmoothZoom(rootDir, vidObj, lowResFactor, highResFactor, orgImageData, center, vidRes)
%% name scheme Defines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lastIm is the last frame from the lower resolution / lower fedelity movie
% firstIm is the first frame from the higher resolution / higher fedelity
% movie
% bigIm is a larger version of lastIm (interpolated version using the
% renderer)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Image reading
dbstop('in','makeSmoothZoom.m','at','30');

lowDir = dir(fullfile(rootDir,sprintf('x%d',lowResFactor),'ScreenShots','*.bmp'));
lastImData = readMetaData(fullfile(rootDir,sprintf('x%d',lowResFactor)));
lastIm = imread(fullfile(rootDir,sprintf('x%d',lowResFactor),'ScreenShots',lowDir(end).name));

highDir = dir(fullfile(rootDir,sprintf('x%d',highResFactor),'ScreenShots','*.bmp'));
firstImData = readMetaData(fullfile(rootDir,sprintf('x%d',highResFactor)));
firstIm = imread(fullfile(rootDir,sprintf('x%d',highResFactor),'ScreenShots',highDir(end).name));

bigDir = dir(fullfile(rootDir,'big',sprintf('x%d_*.bmp',lowResFactor)));
bigIm = imread(fullfile(rootDir,'big',bigDir(1).name));

highResFactor = max(highResFactor,1);

%% find regions of interests (cut out the volumes from the backgrounds)
newWay = 0;

subtractBorder = 10;
blackIdx = bigIm==0;
stats = regionprops(blackIdx(:,:,1),'BoundingBox','Area');
bigRoi = [floor(stats(1).BoundingBox(1)+subtractBorder),...
           floor(stats(1).BoundingBox(2)+subtractBorder),...
           floor(stats(1).BoundingBox(1))+stats(1).BoundingBox(3)-subtractBorder*2+1,...
           floor(stats(1).BoundingBox(2))+stats(1).BoundingBox(4)-subtractBorder*2+1];
bigRoiIm = bigIm(bigRoi(2):bigRoi(4), bigRoi(1):bigRoi(3), :);

blackIdx = lastIm==0;
stats = regionprops(blackIdx(:,:,1),'BoundingBox','Area');
lastRoi = [floor(stats(1).BoundingBox(1)+subtractBorder),...
           floor(stats(1).BoundingBox(2)+subtractBorder),...
           floor(stats(1).BoundingBox(1))+stats(1).BoundingBox(3)-subtractBorder*2+1,...
           floor(stats(1).BoundingBox(2))+stats(1).BoundingBox(4)-subtractBorder*2+1];
lastRoiIm = lastIm(lastRoi(2):lastRoi(4), lastRoi(1):lastRoi(3), :);

backGroundIm = lastIm;
backGroundIm(floor(stats(1).BoundingBox(2)):floor(stats(1).BoundingBox(4))+floor(stats(1).BoundingBox(2)),...
    floor(stats(1).BoundingBox(1)):floor(stats(1).BoundingBox(3))+floor(stats(1).BoundingBox(1)), :) = ...
    ones(floor(stats(1).BoundingBox(4))+1,floor(stats(1).BoundingBox(3))+1, 3,'uint8').*lastIm(1);

blackIdx = firstIm==0;
stats = regionprops(blackIdx(:,:,1),'BoundingBox','Area');
firstRoi = [floor(stats(1).BoundingBox(1)+subtractBorder),...
           floor(stats(1).BoundingBox(2)+subtractBorder),...
           floor(stats(1).BoundingBox(1))+stats(1).BoundingBox(3)-subtractBorder*2+1,...
           floor(stats(1).BoundingBox(2))+stats(1).BoundingBox(4)-subtractBorder*2+1];
firstRoiIm = firstIm(firstRoi(2):firstRoi(4), firstRoi(1):firstRoi(3), :);

%% calculate zoom box
firstImSzInOrg = [firstImData.YDimension*highResFactor firstImData.XDimension*highResFactor];
lastImSzInOrg = [lastImData.YDimension*lowResFactor lastImData.XDimension*lowResFactor];

firstsCoorInOrg = [floor(center(1)-firstImSzInOrg(2)/2),...
                      floor(center(2)-firstImSzInOrg(1)/2),...
                      floor(center(1)+firstImSzInOrg(2)/2),...
                      floor(center(2)+firstImSzInOrg(1)/2)];
                  
bigsCoorInOrg = [floor(center(1)-lastImSzInOrg(2)/2),...
                      floor(center(2)-lastImSzInOrg(1)/2),...
                      floor(center(1)+lastImSzInOrg(2)/2),...
                      floor(center(2)+lastImSzInOrg(1)/2)];
                  
if (firstsCoorInOrg(1)<1)
    % zoom was railed on left
    firstsCoorInOrg(1) = 1;
    firstsCoorInOrg(3) = min(orgImageData.XDimension, firstImSzInOrg(2));
end

if (firstsCoorInOrg(2)<1)
    % zoom was railed on the top
    firstsCoorInOrg(2) = 1;
    firstsCoorInOrg(4) = min(orgImageData.YDimension, firstImSzInOrg(1));
end

if (firstsCoorInOrg(3)>orgImageData.XDimension)
    % zoom was railed on right
    firstsCoorInOrg(3) = orgImageData.XDimension;
    firstsCoorInOrg(1) = orgImageData.XDimension-firstImSzInOrg(2)+1;%if this is less than 1 it is an error
end

if (firstsCoorInOrg(4)>orgImageData.YDimension)
    % zoom was railed on the top
    firstsCoorInOrg(4) = orgImageData.YDimension;
    firstsCoorInOrg(2) = orgImageData.YDimension-firstImSzInOrg(1)+1;%if this is less than 1 it is an error
end

if (bigsCoorInOrg(1)<1)
    % zoom was railed on left
    bigsCoorInOrg(1) = 1;
    bigsCoorInOrg(3) = min(orgImageData.XDimension, lastImSzInOrg(2));
end

if (bigsCoorInOrg(2)<1)
    % zoom was railed on the top
    bigsCoorInOrg(2) = 1;
    bigsCoorInOrg(4) = min(orgImageData.YDimension, lastImSzInOrg(1));
end

if (bigsCoorInOrg(3)>orgImageData.XDimension)
    % zoom was railed on right
    bigsCoorInOrg(3) = orgImageData.XDimension;
    bigsCoorInOrg(1) = orgImageData.XDimension-lastImSzInOrg(2)+1;%if this is less than 1 it is an error
end

if (bigsCoorInOrg(4)>orgImageData.YDimension)
    % zoom was railed on the top
    bigsCoorInOrg(4) = orgImageData.YDimension;
    bigsCoorInOrg(2) = orgImageData.YDimension-lastImSzInOrg(1)+1;%if this is less than 1 it is an error
end

vox2pX = size(bigRoiIm,2)/lastImData.XDimension;
vox2pY = size(bigRoiIm,1)/lastImData.YDimension;

if (newWay==1)
    dbstop('in','makeSmoothZoom.m','at','140');
    boxPos = [];
    %boxPos([1,3]) = boxPos([1,3]) + ;
    %boxPos(3) = boxPos(3) + ;
    %boxPos([2,4]) = boxPos([2,4]) + ;
    %boxPos(4) = boxPos(4) + ;
else
    boxPos([1,2]) = firstsCoorInOrg([1,2]) - bigsCoorInOrg([1,2]);
    boxPos([3,4]) = [firstImSzInOrg(2) firstImSzInOrg(1)];
    boxPos = boxPos./lowResFactor; % convert to lowRes (lastIm) space
    boxPos([1,3]) = floor(boxPos([1,3]).*vox2pX); % slight factor when resizeing in renderer
    boxPos([2,4]) = floor(boxPos([2,4]).*vox2pY);
end

xZoomStart = [lastRoi(1),lastRoi(3)-lastRoi(1)+1];
yZoomStart = [lastRoi(2),lastRoi(4)-lastRoi(2)+1];
xZoomEnd = [firstRoi(1),firstRoi(3)-firstRoi(1)+1];
yZoomEnd = [firstRoi(2),firstRoi(4)-firstRoi(2)+1];

xCropStart = [1,size(bigRoiIm,2)];
yCropStart = [1,size(bigRoiIm,1)];
xCropEnd = [boxPos(1),boxPos(1)+boxPos(3)];
yCropEnd = [boxPos(2),boxPos(2)+boxPos(4)];

fig = figure;
if (newWay==1)
    imPut = imresize(firstRoiIm(1:end,1:end,:),[yCropEnd(2)-yCropEnd(1)+1,xCropEnd(2)-xCropEnd(1)+1]);
    bigRoiImW = bigRoiIm;
    bigRoiImW(yCropEnd(1):yCropEnd(2),xCropEnd(1):xCropEnd(2),:) = imPut;
    imshow(bigRoiImW);
    %figure, imshow(bigRoiIm)
else
    imshow(bigRoiIm);
end

hold on
rectangle('Position',boxPos,'EdgeColor','w');

set(gcf,'Position',[100 100 1920 1080])
set(gca,'Position',[0 0 1 1]);

imageSeq = 1;
n=60;
xCropInc = (xCropEnd-xCropStart) / (n-1);
yCropInc = (yCropEnd-yCropStart) / (n-1);
xZoomInc = (xZoomEnd-xZoomStart) / (n-1);
yZoomInc = (yZoomEnd-yZoomStart) / (n-1);

for i=0:n-1
    xlim(xCropStart + i*xCropInc);
    ylim(yCropStart + i*yCropInc);
    
    placeX = round(xZoomStart + i*xZoomInc);
    placeY = round(yZoomStart + i*yZoomInc);
    
    imageData = getframe(gca);
    imLowPlace = imresize(imageData.cdata(:,2:end-1,:),[placeY(2) placeX(2)]);
    
    curIm = backGroundIm;
    curIm(placeY(1):placeY(1)+placeY(2)-1,placeX(1):placeX(1)+placeX(2)-1,:) = imLowPlace;
    curIm = imresize(curIm,vidRes);
    
    %imwrite(curIm,fullfile(rootDir,sprintf('%s_%dx-%dx_%d.tif',orgImageData.DatasetName,lowResFactor,highResFactor,i)),'tif','Compression','lzw');
    
    if (i==0)
        for j=1:14
            imageSeq = imageSeq +1;
            writeVideo(vidObj,im2frame(curIm));
        end
    end
    
    imageSeq = imageSeq +1;
    writeVideo(vidObj,im2frame(curIm));
end

close(fig)
end