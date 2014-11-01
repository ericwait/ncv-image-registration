stains = struct('stain','','color',[]);
stains(1).stain = 'DAPI';
stains(1).color = [0.00, 0.00, 1.00];
stains(2).stain = 'Olig2';
stains(2).color = [0.75, 0.75, 0.00];
stains(3).stain = 'AcTub';
stains(3).color = [0.75, 0.75, 0.00];
stains(4).stain = 'GFAP';
stains(4).color = [0.00 1.00 0.00];
stains(5).stain = 'Dcx';
stains(5).color = [0.00 1.00 1.00];
stains(6).stain = 'VCAM';
stains(6).color = [1.00 0.00 1.00];
stains(7).stain = 'Laminin';
stains(7).color = [1.00 0.00 0.00];
stains(8).stain = 'laminin';
stains(8).color = [1.00 0.00 0.00];
stains(9).stain = 'Bcatenin';
stains(9).color = [1.00 1.00 0.00];
stains(10).stain = 'Mash';
stains(10).color = [1.00 0.00 1.00];
stains(11).stain = 'NCAM';
stains(11).color = [0.00 1.00 1.00];
stains(12).stain = 'EDU';
stains(12).color = [1.00, 0.00, 0.00];
stains(13).stain = 'DCX';
stains(13).color = [0.00 1.00 1.00];

[fileName,root,~] = uigetfile('.txt');
imageData = readMetaData(fullfile(root,fileName));
colors = zeros(1,1,3,imageData.NumberOfChannels);

starts = zeros(1,length(stains));
for i=1:length(stains)
    idx = strfind(imageData.DatasetName,stains(i).stain);
    if (~isempty(idx))
        starts(i) = idx;
    end
end

[b, idx] = sort(starts);
stainOrder = idx(b>0);
if (isempty(stainOrder) || length(stainOrder)~=imageData.NumberOfChannels)
    dbstop in colorMIP at 44
end
for c=1:imageData.NumberOfChannels
    colors(1,1,:,c) = stains(stainOrder(c)).color;
end

imColors = zeros(imageData.YDimension,imageData.XDimension,3,imageData.NumberOfChannels);
imIntensity = zeros(imageData.YDimension,imageData.XDimension,imageData.NumberOfChannels);
for c=1:imageData.NumberOfChannels
    imIntensity(:,:,c) = mat2gray(imread(fullfile(root,sprintf('_%s_c%02d_t0001.tif',imageData.DatasetName,c))));
    color = repmat(colors(1,1,:,c),imageData.YDimension,imageData.XDimension,1);
    imColors(:,:,:,c) = repmat(imIntensity(:,:,c),1,1,3).*color;
end

imMax = max(imIntensity,[],3);
imIntSum = sum(imIntensity,3);
imIntSum(imIntSum==0) = 1;
imColrSum = sum(imColors,4);
imFinal = imColrSum.*repmat(imMax./imIntSum,1,1,3);
fileName = fullfile(root,sprintf('_%s.tif',imageData.DatasetName));
imwrite(imFinal,fileName,'tif','Compression','lzw');