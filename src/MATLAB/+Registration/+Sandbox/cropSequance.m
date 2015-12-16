function imOut = cropSequance(imIn, cropSize, imMid)
imSize = size(imIn);

if (~exist('imMid','var') || isempty(imMid))
    imMid = floor(imSize/2);
end

cropRadius = floor(cropSize/2);

figure;imagesc(max(imIn(:,:,:,1,4),[],3));colormap(gray);hold on;
rectangle('Position',[(imMid(2)-cropRadius(2)+1) (imMid(1)-cropRadius(1)+1) cropSize(2)-1 cropSize(1)-1],'EdgeColor','r');

imOut = imIn(...
    (imMid(1)-cropRadius(1)+1):(imMid(1)+cropRadius(1)),...
    (imMid(2)-cropRadius(2)+1):(imMid(2)+cropRadius(2)),...
    (imMid(3)-cropRadius(3)+1):(imMid(3)+cropRadius(3)),...
    :,:);

figure;imagesc(max(imOut(:,:,:,1,4),[],3));colormap(gray)
end
