[im,imD] = MicroscopeData.ReaderH5('D:\Images\Yu');
showplots = true;

prgs = Utils.CmdlnProgress(imD.Dimensions(3)-1,true,'Flip');

maxSlice = ones(1,3);
minSlice = ones(1,3);
for c=1:3
    curIm = im(:,:,:,c);
    [~,I] = max(curIm(:));
    [~,~,maxSlice(c)] = ind2sub(size(im1),I);
    [~,I] = min(curIm(:));
    [~,~,minSlice(c)] = ind2sub(size(im1),I);
end

imH = im;
for c=1:3
    refIm = im(:,:,maxSlice(c),c);
    for z=1:imD.Dimensions(3)
        curIm = im(:,:,z,c);
        imH(:,:,z,c) = imhistmatch(curIm,refIm,255);
    end
end

onesIm = ones(Utils.SwapXY_RC(imD.Dimensions),'like',im)*255;
imNorm = onesIm - imH(:,:,:,2,:);
imS = imH;

for z=1:imD.Dimensions(3)
    curIm = imNorm(:,:,z,:,:);
    curIm = curIm - min(curIm(:));
    curIm = single(curIm)./single(max(curIm(:)));
    curIm = Cuda.ContrastEnhancement(curIm,[75,75,1],[3,3,1],1);
    curIm = single(curIm)./single(max(curIm(:)));
    imNorm(:,:,z,:,:) = im2uint8(curIm);
    
    prgs.PrintProgress(z);
end
prgs.ClearProgress(true);

imS(:,:,:,2,:) = imNorm;
clear imNorm

imS(:,:,:,1,:) = Cuda.MeanFilter(im(:,:,:,1,:),[5,5,1],1);
imS(:,:,:,3,:) = Cuda.ContrastEnhancement(im(:,:,:,3,:),[75,75,1],[3,3,1],1);

%%
%mkdir('regMovie');
allDeltas_rc = zeros(imD.Dimensions(3),4);
offsets_rc = zeros(size(allDeltas_rc));
allNCV = zeros(imD.Dimensions(3),1);
imDs = imD;
imDs.Dimensions(3) = 1;
imDs.NumberOfChannels = 1;

prgs = Utils.CmdlnProgress(imD.Dimensions(3)-1,false,'Reg');
for z=1:imD.Dimensions(3)-1   
    tic
    curImA = im(:,:,z,:);
    curImB = im(:,:,z+1,:);
    curImBh = imhistmatch(squeeze(curImB),squeeze(curImA),255);
    if (showplots)
        f = figure;
        subplot(1,3,1);
        imshowpair(im(:,:,z,2),im(:,:,z+1,2));
        title('Original');
    end
    deltas_rc = zeros(3,4);
    ncv = zeros(3,1);

    for c=1:3
        imA = curImA(:,:,:,c);
        imB = curImB(:,:,:,c);
    
        [deltas_rc(c,:),ncv(c,:)] = Registration.FFT.GetMaxNCVdeltasRotate(imA,[],imB,[],20,0.5,75^2,150,[],false,c);
    end
    
    [~,I] = max(ncv);
    allDeltas_rc(z+1,:) = deltas_rc(I,:);
    allNCV(z+1) = ncv(I,:);
    offsets_rc(z+1,:) = offsets_rc(z,:) + deltas_rc(I,:);
    
    if (showplots)
        [imAr,~,imBr] = Registration.Sandbox.ApplyDeltasCenter(im(:,:,z,2),im(:,:,z+1,2),[0,0,0,deltas_rc(I,4)]);
        
        subplot(1,3,2)
        imshowpair(imAr,imBr);
        title('Rotated Only');
        
        [imAr,~,imBr] = Registration.Sandbox.ApplyDeltasCenter(im(:,:,z,2),im(:,:,z+1,2),deltas_rc(I,:));
        
        subplot(1,3,3)
        imshowpair(imAr,imBr);
        title(sprintf('\\Delta (%.1fx, %.1fy, %.1f\\circ) = %.4f',deltas_rc(I,2),deltas_rc(I,1),deltas_rc(I,4),ncv(I)));
        set(f,'units','normalized','Position',[0,0,1,1],'Name',sprintf('(%.1fx, %.1fy, %.1fz, %.1f) in %s', offsets_rc(z+1,2), offsets_rc(z+1,1), z+1, offsets_rc(z+1,4),Utils.PrintTime(toc)));
        fr = getframe(f);
        imwrite(fr.cdata,fullfile('regMovie',sprintf('%04d.tif',z)),'compression','lzw');
        close(f);
    end
    
    prgs.PrintProgress(z);
end
prgs.ClearProgress(true);

clear imA
clear imB
clear imBr
clear imAr
%%

offMax = max(abs(offsets_rc),[],1);
xydim = ceil(max(imD.Dimensions(1:2))+ * 2 + sqrt(max(imD.Dimensions(1:2))));
outIm = zeros(xydim,xydim,imD.Dimensions(3),imD.NumberOfChannels,'like',im);
outImCenter = [size(outIm,1),size(outIm,2)]./2;
upperLeft = round(outImCenter - [size(im,1),size(im,2)]./2);

for z=1:imD.Dimensions(3)
    curIm = im(:,:,z,:);
    curImR = imrotate(curIm,offsets_rc(z,4),'bicubic');
    imStart = upperLeft+offsets_rc(z,1:2); % move the upperLeft over by the delta
    outIm(imStart(1):imStart(1)+size(curImR,1)-1,imStart(2):imStart(2)+size(curImR,2)-1,z,:) = curImR;
end

imMax = max(squeeze(max(outIm,[],3)),[],3);
imBW = imMax>0;

rp = regionprops(imBW,'boundingbox');
bb = rp.BoundingBox;
bb(1:2) = floor(bb(1:2));
bb(3:4) = ceil(bb(3:4));

imNew = outIm(bb(1):bb(3),bb(2):bb(4),:,:);

imDNew = imD;
imDNew.Dimensions = Utils.SwapXY_RC(size(imNew));
imDNew.Dimensions = imDNew.Dimensions(1:3);
% 
% for z=2:imD.Dimensions(3)
%     figure
%     imshowpair(outIm(:,:,z-1,2),outIm(:,:,z,2));
% end