function FindChrisPositions(listPath,mipName,backgroundColor,overlap)
%% Check Inputs
if (~exist('listPath','var'))
    listPath = [];
end

% Get subimage metadata
[ imageDatasets, ~ ] = Registration.GetMontageSubMeta(listPath);

if (~exist('mipName','var') || isempty(mipName))
    [mipName,root,~] = uigetfile(fullfile(imageDatasets(1).imageDir,'..','.tif'));
end

if (isempty(imageDatasets) || isempty(mipName) || any(mipName==0))
    warning('No files found, exiting');
    return;
end

if (~exist('overlap','var') || isempty(overlap))
    overlap = 0.10;
end

if (~exist('backgroundColor','var') || isempty(bacgroundColor))
    backgroundColor = [69,77,98];
end

%% Read in template
imMont = imread(fullfile(root,mipName));

%% Mask out the background color
mask = ~(imMont(:,:,1)==backgroundColor(1) &...
    imMont(:,:,2)==backgroundColor(2) &...
    imMont(:,:,3)==backgroundColor(3));

%% Find the montage area
cc = bwconncomp(mask);

%% Calculate the average side length in the square images
numImages = length(imageDatasets);
subImageLength = ceil(sqrt(size(cc.PixelIdxList{1},1)/numImages));

%% Show the original image
figure
imagesc(imMont)
axis image
hold on

%% Draw the partition lines
for i=subImageLength:subImageLength:size(imMont,2)
    line([i,i],[0,size(imMont,1)],'color','w');
end

for i=subImageLength:subImageLength:size(imMont,1)
    line([0,size(imMont,2)],[i,i],'color','w');
end

%% Calculate the index of each box
% Index should be the following:
% 1->2->3
%       |
%       V
% 6<-5<-4
% |
% V
% 7->8->9 and so on

pos_RC = zeros(numImages,2);
imNum = 1;

% Search each row to the right then to the left, thus stride by 2
for r=1:2:ceil(size(imMont,1)/subImageLength)
    % Find the center of the current row
    rowCenter = r*subImageLength-floor(subImageLength/2);
    
    if (size(imMont,1)<rowCenter)
        % Too far
        continue;
    end
    
    for c=1:ceil(size(imMont,2)/subImageLength)
        % Find the center of the current column starting on the left
        colCenter = c*subImageLength-floor(subImageLength/2);
        
        if (size(imMont,2)<colCenter)
            % Too far
            continue;
        end
        
        if (mask(rowCenter,colCenter))
            % This center is in the montage
            pos_RC(imNum,:) = [r,c];
            text(colCenter,rowCenter,num2str(imNum),'color','w');
            imNum = imNum + 1;
        end
    end
    
    % Go to the next row
    rowCenter = (r+1)*subImageLength-floor(subImageLength/2);
    
    if (size(imMont,1)<rowCenter)
        % Too Far
        continue;
    end
    
    for c=ceil(size(imMont,2)/subImageLength):-1:1
        % Find the center of the current column starting on the right
        colCenter = c*subImageLength-floor(subImageLength/2);
        
        if (size(imMont,2)<colCenter)
            % Too Far
            continue;
        end
        
        if (mask(rowCenter,colCenter))
            % This center is in the montage
            pos_RC(imNum,:) = [r+1,c];
            text(colCenter,rowCenter,num2str(imNum),'color','w');
            imNum = imNum + 1;
        end
    end
end
hold off

%% Write the positions back to the metadata
for i=1:length(imageDatasets)
    d = imageDatasets(i);
    
    numIdx = regexp(imageDatasets(i).DatasetName,'#\d') +1;
    numStr = imageDatasets(i).DatasetName(numIdx:end);
    n = str2double(numStr);
    % Make the positions start at 0
    curPos_RC = pos_RC(n,:)-1;
    
    d.XPosition = curPos_RC(2) * (d.XDimension-d.XDimension*overlap) * d.XPixelPhysicalSize * 1e-5;
    d.YPosition = curPos_RC(1) * (d.YDimension-d.YDimension*overlap) * d.YPixelPhysicalSize * 1e-5;
    d.ZPosition = 0;
    
    % Write data back
    MicroscopeData.CreateMetadata(d.imageDir,d,true);
end
end
