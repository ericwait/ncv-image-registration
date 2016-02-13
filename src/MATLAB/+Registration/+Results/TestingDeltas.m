function [fig,ax] = TestingDeltas(outImage, outImageColor,imageDatasets, minPos_XY, chan,curDataset)
[img, colorIdx] = max(outImage,[],3);

imageHandle = ImUtils.ThreeD.ShowMaxImage(img,true,[],[],true);
ax = get(imageHandle,'Parent');
fig = get(ax,'Parent');

Registration.Results.DrawBoxesLines(fig,ax,imageDatasets,minPos_XY,chan,curDataset);

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
    Registration.Results.DrawBoxesLines(fig,ax,imageDatasets,minPos_XY,chan,curDataset);
end

drawnow
end
