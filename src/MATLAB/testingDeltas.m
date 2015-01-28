function [fig,ax] = testingDeltas(outImage, outImageColor,imageDatasets,chan)
[img, colorIdx] = max(outImage,[],3);

fig = figure;
imagesc(img);
ax = gca;
colormap(ax,'gray');

drawBoxesLines(fig,ax,imageDatasets,chan);

if (~isempty(outImageColor))
    %% setup colormap for the colored image
    [r,c] = ndgrid(1:size(outImage,1),1:size(outImage,2));
    idx = sub2ind(size(outImage),r,c,colorIdx);
    pixelColors = outImageColor(idx);
    cmap = hsv(12);
    cmap = cmap(randi(12, max(double(pixelColors(:)))+1),:);
    alpha = cat(3, reshape(cmap(pixelColors+1,1),size(pixelColors)), reshape(cmap(pixelColors+1,2),size(pixelColors)), reshape(cmap(pixelColors+1,3),size(pixelColors)));
    imgC = alpha .* repmat(mat2gray(img),[1 1 3]);
    
    fig = figure;
    image(imgC);
    ax = gca;
    drawBoxesLines(fig,ax,imageDatasets,chan);
end

drawnow

end
