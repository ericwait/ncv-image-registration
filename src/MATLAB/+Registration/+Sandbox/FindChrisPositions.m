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
subImageLength_rc = [subImageLength,subImageLength];

%% Show the original image
f = figure;
ax = imagesc(imMont);
axis image
hold on

%% Draw the partition lines
for i=subImageLength_rc(2):subImageLength_rc(2):size(imMont,2)
    line([i,i],[0,size(imMont,1)],'color','w');
end

for i=subImageLength_rc(1):subImageLength_rc(1):size(imMont,1)
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
for r=1:2:ceil(size(imMont,1)/subImageLength_rc(1))
    % Find the center of the current row
    rowCenter = r*subImageLength_rc(1)-floor(subImageLength_rc(1)/2);
    
    if (size(imMont,1)<rowCenter)
        % Too far
        continue;
    end
    
    for c=1:ceil(size(imMont,2)/subImageLength_rc(2))
        % Find the center of the current column starting on the left
        colCenter = c*subImageLength_rc(2)-floor(subImageLength_rc(2)/2);
        
        if (size(imMont,2)<colCenter)
            % Too far
            continue;
        end
        
        if (mask(round(rowCenter),round(colCenter)))
            % This center is in the montage
            pos_RC(imNum,:) = [r,c];
            text(colCenter,rowCenter,num2str(imNum),'color','w');
            imNum = imNum + 1;
        end
    end
    
    % Go to the next row
    rowCenter = (r+1)*subImageLength_rc(1)-floor(subImageLength_rc(1)/2);
    
    if (size(imMont,1)<rowCenter)
        % Too Far
        continue;
    end
    
    for c=ceil(size(imMont,2)/subImageLength_rc(2)):-1:1
        % Find the center of the current column starting on the right
        colCenter = c*subImageLength_rc(2)-floor(subImageLength_rc(2)/2);
        
        if (size(imMont,2)<colCenter)
            % Too Far
            continue;
        end
        
        if (mask(round(rowCenter),round(colCenter)))
            % This center is in the montage
            pos_RC(imNum,:) = [r+1,c];
            text(colCenter,rowCenter,num2str(imNum),'color','w');
            imNum = imNum + 1;
        end
    end
end
hold off
set(f,'units','normalized','Position',[0,0,1,1]);
set(gca,'units','normalized','Position',[0,0,1,1]);

frm = getframe(gca);
imwrite(frm.cdata,fullfile(imageDatasets(1).imageDir,'..',sprintf('_%s_pos.tif',imageDatasets(1).DatasetName)),'tif','Compression','lzw');

close(f);

%% Write the positions back to the metadata
for i=1:length(imageDatasets)
    d = imageDatasets(i);
    
    numIdx = regexp(imageDatasets(i).DatasetName,'_pos\d') +1;
    numStr = imageDatasets(i).DatasetName(numIdx+3:end);
    n = str2double(numStr);
    % Make the positions start at 0
    curPos_RC = pos_RC(n,:)-1;
    
    XPosition = curPos_RC(2) * (d.Dimensions(1)-d.Dimensions(1)*overlap) * d.PixelPhysicalSize(1) * 1e-6;
    YPosition = curPos_RC(1) * (d.Dimensions(2)-d.Dimensions(2)*overlap) * d.PixelPhysicalSize(2) * 1e-6;
    ZPosition = 0;
    
    d.Position = [XPosition,YPosition,ZPosition];
    [ colors, stainNames ] = MicroscopeData.Colors.GetChannelColors( d, false );
    d.ChannelNames = stainNames;
    d.ChannelColors = colors;
    
    % Write data back
    MicroscopeData.CreateMetadata(d.imageDir,d,true);
end
end
