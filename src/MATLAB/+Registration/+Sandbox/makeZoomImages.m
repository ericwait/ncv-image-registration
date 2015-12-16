function makeZoomImages()

%% set these before running
rootDir = 'D:\Users\Eric\Documents\Images\22mo2 wmSVZ Unmixed\DAPI Olig2-514 GFAP-488 Mash1-647 PSA-NCAM-549 lectin-568 22mo wmSVZ_Montage_wDelta';
center = [7953 1992]; %[x y]

%% run
outImSize = [576 1024]-2;

reduc = 1;
imagesPath = fullfile(rootDir,['x' num2str(reduc)]);

[im, imageData] = tiffReader('uint8',[],[],[],imagesPath,false);
curCenter = round(center/reduc);
rowMin = max(1,curCenter(2)-outImSize(1)/2);
rowMax = min(size(im,1),curCenter(2)+outImSize(1)/2);
colMin = max(1,curCenter(1)-outImSize(2)/2);
colMax = min(size(im,2),curCenter(1)+outImSize(2)/2);

imROI = im(rowMin:rowMax,colMin:colMax,:,:,:);

if (~exist(fullfile(rootDir,'movie','x0'),'file'))
    mkdir(fullfile(rootDir,'movie','x0'));
end

imageData.YDimension = colMax-colMin+1;
imageData.XDimension = rowMax-rowMin+1;
createMetadata(fullfile(rootDir,'movie','x0'),imageData);

tiffWriter(imROI,[fullfile(rootDir,'movie','x0') '\' imageData.DatasetName]);

maxReduction = ceil(max(size(im))/1024);

clear im;
clear imROI;

outImSize = [1152 2048]-2; %[row col]

for reduc=1:maxReduction
    imagesPath = fullfile(rootDir,['x' num2str(reduc)]);
    
    [im, imageData] = tiffReader('uint8',[],[],[],imagesPath,false);
    curCenter = round(center/reduc);
    rowMin = max(1,curCenter(2)-outImSize(1)/2);
    rowMax = min(size(im,1),curCenter(2)+outImSize(1)/2);
    colMin = max(1,curCenter(1)-outImSize(2)/2);
    colMax = min(size(im,2),curCenter(1)+outImSize(2)/2);
    
    imROI = im(rowMin:rowMax,colMin:colMax,:,:,:);
    
    if (~exist(fullfile(rootDir,'movie',['x' num2str(reduc)]),'file'))
        mkdir(fullfile(rootDir,'movie',['x' num2str(reduc)]));
    end
    
    imageData.YDimension = colMax-colMin+1;
    imageData.XDimension = rowMax-rowMin+1;
    createMetadata(fullfile(rootDir,'movie',['x' num2str(reduc)]),imageData);
    
    tiffWriter(imROI,[fullfile(rootDir,'movie',['x' num2str(reduc)]) '\' imageData.DatasetName]);
    
    clear im;
    clear imROI;
end
end
