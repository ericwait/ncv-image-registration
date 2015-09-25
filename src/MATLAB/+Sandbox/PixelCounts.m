function PixelCounts(path,chans)
tic
[im,imData] = tiffReader(path,[],chans);
fprintf('Read images in: %s\n',printTime(toc));

if (isempty(im))
    return
end

funcTime = tic;
[~,~,maxVal] = classBits(im);

imBWmat = zeros(size(im),'like',im);
imBW = zeros(size(im),'like',im);
sumsMat = zeros(1,size(im,4));
sums = zeros(1,size(im,4));
[~, memoryStats] = CudaMex('DeviceCount');
[~,device] = max([memoryStats.available]);
figure
for c = 1:length(chans)
    imLcl = threshNonZeros(im(:,:,:,c,:),maxVal);
    sumsMat(c) = sum(imLcl(:));
    imBWmat(:,:,:,c,:) = imLcl;
    
    cudaTime = tic;
    imLcl = CudaMex('ContrastEnhancement',im(:,:,:,c,:),[250,250,100],[3,3,3],device);
    fprintf('Cuda took: %s\n',printTime(toc(cudaTime)));
    
    imLcl = threshNonZeros(imLcl,maxVal);
    sums(c) = sum(imLcl(:));
    imBW(:,:,:,c,:) = imLcl;
    
    subplot(3,length(chans),c)
    imagesc(max(im(:,:,:,c,:),[],3))
    colormap gray
    axis image
    title([imData.DatasetName ' Channel:' num2str(c)]);
    
    subplot(3,length(chans),c+length(chans))
    imagesc(max(imBWmat(:,:,:,c,:),[],3))
    colormap gray
    axis image
    title(['Number of pixels:' num2str(sumsMat(c)) ' Of DAPI:' num2str(sumsMat(c)/sumsMat(1)*100) '%']);
    
    subplot(3,length(chans),c+2*length(chans))
    imagesc(max(imBW(:,:,:,c,:),[],3))
    colormap gray
    axis image
    title(['Number of pixels:' num2str(sums(c)) ' Of DAPI:' num2str(sums(c)/sums(1)*100) '%']);
    drawnow
end
fprintf('Completed in: %s\n',printTime(toc(funcTime)));
end

function imBW = threshNonZeros(im,max)
    imNZ = im(im>0);
    thresh = graythresh(imNZ(:))*max;
    imBW = im>thresh;
end
