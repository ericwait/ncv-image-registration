[im,imD] = MicroscopeData.ReaderH5('D:\Images\Yu');
showplots = true;

maxSlice = ones(1,3);
minSlice = ones(1,3);
for c=1:3
    curIm = im(:,:,:,c);
    [~,I] = max(curIm(:));
    [~,~,maxSlice(c)] = ind2sub(size(curIm),I);
    [~,I] = min(curIm(:));
    [~,~,minSlice(c)] = ind2sub(size(curIm),I);
end

prgs = Utils.CmdlnProgress(imD.NumberOfChannels-1,true,'histo');
imH = im;
for c=1:3
    refIm = im(:,:,maxSlice(c),c);
    for z=1:imD.Dimensions(3)
        curIm = im(:,:,z,c);
        imH(:,:,z,c) = imhistmatch(curIm,refIm,255);
    end
    prgs.PrintProgress(c);
end
prgs.ClearProgress(true);

onesIm = ones(Utils.SwapXY_RC(imD.Dimensions),'like',im)*255;
imNorm = onesIm - imH(:,:,:,2,:);
imS = imH;

prgs = Utils.CmdlnProgress(imD.Dimensions(3)-1,true,'Flip');
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
[optimizer,metric] = imregconfig('monomodal');
imR = im;
prgs = Utils.CmdlnProgress((imD.Dimensions(3)-1)*imD.NumberOfChannels,true,'Reg');
for c=1:imD.NumberOfChannels
    for z=2:imD.Dimensions(3)
        imR(:,:,z,c) = imregister(im(:,:,z,c),im(:,:,z-1,c),'affine',optimizer,metric);
        prgs.PrintProgress((c-1)*imD.Dimensions(3)+z);
    end
end
prgs.ClearProgress(true);
