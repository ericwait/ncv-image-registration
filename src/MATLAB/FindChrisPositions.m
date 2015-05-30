function FindChrisPositions(root,mipName)
if (~exist('root','var') || isempty(root))
    root = uigetdir();
end

if (~exist('mipName','var') || isempty(mipName))
    mipName = uigetfile(fullfile(root,'.tif'));
end

if (isempty(root) || isempty(mipName) || any(mipName==0))
    warning('No files found, exiting');
    return;
end

imMont = imread(fullfile(root,mipName));
overlap = 0.05;

[listFile,err] = fopen(fullfile(root,'_unmixed','list.txt'),'r');

if (~isempty(err))
    warning(err);
    return;
end

names = textscan(listFile,'%s','delimiter','\n');
fclose(listFile);

txtNames = names{1,1};
numImages = length(txtNames);

mask = ~(imMont(:,:,1)==69 & imMont(:,:,2)==77 & imMont(:,:,3)==98);
cc = bwconncomp(mask);

edgeSize = ceil(sqrt(size(cc.PixelIdxList{1},1)/numImages));

figure
imagesc(imMont)
axis image
hold on

for i=edgeSize:edgeSize:size(imMont,2)
    line([i,i],[0,size(imMont,1)],'color','w');
end

for i=edgeSize:edgeSize:size(imMont,1)
    line([0,size(imMont,2)],[i,i],'color','w');
end

pos = zeros(numImages,2);
imNum = 1;
for r=1:2:ceil(size(imMont,1)/edgeSize)
    y = r*edgeSize-floor(edgeSize/2);
    
    if (size(imMont,1)<y)
        continue;
    end
    
    for c=1:ceil(size(imMont,2)/edgeSize)
        x = c*edgeSize-floor(edgeSize/2);
        
        if (size(imMont,2)<x)
            continue;
        end
        
        if (mask(y,x))
            pos(imNum,:) = [r,c];
            text(x,y,num2str(imNum),'color','w');
            imNum = imNum + 1;
        end
    end
    
    y = (r+1)*edgeSize-floor(edgeSize/2);
    
    if (size(imMont,1)<y)
        continue;
    end
    
    for c=ceil(size(imMont,2)/edgeSize):-1:1
        x = c*edgeSize-floor(edgeSize/2);
        
        if (size(imMont,2)<x)
            continue;
        end
        
        if (mask(y,x))
            pos(imNum,:) = [r+1,c];
            text(x,y,num2str(imNum),'color','w');
            imNum = imNum + 1;
        end
    end
end
hold off

for i=1:length(txtNames)
    d = readMetadata(fullfile(root,'_unmixed',txtNames{i}));
    d.XPosition = (pos(i,1)-1)*(d.XDimension-d.XDimension*overlap)*d.XPixelPhysicalSize * 1e-6;
    d.YPosition = (pos(i,2)-1)*(d.YDimension-d.YDimension*overlap)*d.YPixelPhysicalSize * 1e-6;
    createMetadata(fullfile(root,'_unmixed'),d,true);
end
end
