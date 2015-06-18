function imFinal = colorMIP(metadataFilePath,chanList)
if (~exist('chanList','var'))
    chanList = [];
end

%% setup colors
defaultColors = struct('str','','color',[]);
defaultColors(1).str = 'r';
defaultColors(1).color = [1.00 0.00 0.00];
defaultColors(2).str = 'g';
defaultColors(2).color = [0.00 1.00 0.00];
defaultColors(3).str = 'b';
defaultColors(3).color = [0.00 0.00 1.00];
defaultColors(4).str = 'c';
defaultColors(4).color = [0.00 1.00 1.00];
defaultColors(5).str = 'm';
defaultColors(5).color = [1.00 0.00 1.00];
defaultColors(6).str = 'y';
defaultColors(6).color = [1.00 1.00 0.00];

stains = setColors();

%% get image data
if (~exist('metadataFilePath','var') || isempty(metadataFilePath))
    [metadataFileName,root,~] = uigetfile('.txt');
    metadataFilePath = fullfile(root,metadataFileName);
end

imageData = readMetadata(metadataFilePath);

if ~isfield(imageData,'DatasetName'), return, end

%% set stain colors
starts = zeros(1,length(stains));
for i=1:length(stains)
    idx = strfind(imageData.imageDir,stains(i).stain);
    if (~isempty(idx))
        starts(i) = idx;
    end
end

[b, idx] = sort(starts);
stainOrder = idx(b>0);
if (isempty(stainOrder) || length(stainOrder)~=imageData.NumberOfChannels)
    dbstop in colorMIP at 46
    disp([stains(stainOrder).stain]);
end

if (isempty(chanList))
    chanList = 1:imageData.NumberOfChannels;
end

[unusedColors, idx] = setdiff([defaultColors.str],[stains(stainOrder(chanList)).strColor]);
if (~isempty(unusedColors) && length(unusedColors)>6-imageData.NumberOfChannels)
    unused = 1;
    for c=1:length(chanList)-1
        for i=c+1:length(chanList)
            if (strcmp(stains(stainOrder(chanList(c))).strColor,stains(stainOrder(chanList(i))).strColor)~=0)
                stains(stainOrder(chanList(i))).strColor = defaultColors(idx(unused)).str;
                stains(stainOrder(chanList(i))).color = defaultColors(idx(unused)).color;
                unused = unused + 1;
            end
        end
    end
end

colors = zeros(1,1,3,length(chanList));
for c=1:length(chanList)
    colors(1,1,:,c) = stains(stainOrder(chanList(c))).color;
end

%% make colored image
imColors = zeros(imageData.YDimension,imageData.XDimension,3,length(chanList));
imIntensity = zeros(imageData.YDimension,imageData.XDimension,length(chanList));
for c=1:length(chanList)
    imIntensity(:,:,c) = mat2gray(max(tiffReader(metadataFilePath,[],chanList(c),[],[],[],1),[],3));
    color = repmat(colors(1,1,:,c),imageData.YDimension,imageData.XDimension,1);
    imColors(:,:,:,c) = repmat(imIntensity(:,:,c),1,1,3).*color;
end

imMax = max(imIntensity,[],3);
imIntSum = sum(imIntensity,3);
imIntSum(imIntSum==0) = 1;
imColrSum = sum(imColors,4);
imFinal = imColrSum.*repmat(imMax./imIntSum,1,1,3);
[root,~,~] = fileparts(metadataFilePath);
metadataFileName = fullfile(root,sprintf('_%s.tif',imageData.DatasetName));
imwrite(imFinal,metadataFileName,'tif','Compression','lzw');
fprintf('%s\nColors:%s\n',metadataFileName,[stains(stainOrder).strColor]);
winopen(metadataFileName);
end

function stains = setColors()
stains = struct('stain','','color',[],'strColor','');

%% blue
stains(1).stain = 'DAPI';
stains(end).color = [0.00, 0.00, 0.50];
stains(end).strColor = 'b';

%% red
lclColor = [1.00 0.00 0.00];
lclStr = 'r';
stains = setNextColor(stains, 'Laminin', lclColor, lclStr);
stains = setNextColor(stains, 'laminin', lclColor, lclStr);
stains = setNextColor(stains, 'Tomato', lclColor, lclStr);
stains = setNextColor(stains, 'Bcat', lclColor, lclStr);
stains = setNextColor(stains, 'Mash', lclColor, lclStr);
stains = setNextColor(stains, 'lectin', lclColor, lclStr);
stains = setNextColor(stains, 'EdU', lclColor, lclStr);
stains = setNextColor(stains, 'EDU', lclColor, lclStr);

%% green
lclColor = [0.00 1.00 0.00];
lclStr = 'g';
stains = setNextColor(stains, 'GFAP', lclColor, lclStr);
stains = setNextColor(stains, 'NCAM', lclColor, lclStr);
stains = setNextColor(stains, 'VCAM', lclColor, lclStr);

%% cyan
lclColor = [0.00 1.00 1.00];
lclStr = 'c';
stains = setNextColor(stains, 'DCX', lclColor, lclStr);
stains = setNextColor(stains, 'Dcx', lclColor, lclStr);
stains = setNextColor(stains, 'Itga', lclColor, lclStr);
%stains = setNextColor(stains, 'NCAM', lclColor, lclStr);

%% yellow
lclColor = [1.00 1.00 0.00];
lclStr = 'y';
stains = setNextColor(stains, 'Olig2', lclColor, lclStr);
stains = setNextColor(stains, 'EGFR', lclColor, lclStr);
% stains = setNextColor(stains, 'AcTub', lclColor, lclStr);
% stains = setNextColor(stains, 'Bcatenin', lclColor, lclStr);

%% magenta
lclColor = [1.00 0.00 1.00];
lclStr = 'm';
stains = setNextColor(stains, 'AcTub', lclColor, lclStr);
% stains = setNextColor(stains, 'VCAM', lclColor, lclStr);
% stains = setNextColor(stains, 'Mash', lclColor, lclStr);
end

function stains = setNextColor(stains, stainName, val, strC)
stains(end+1).stain = stainName;
stains(end).color = val;
stains(end).strColor = strC;
end
