function cost = SegmentationNCV(im1,seg1Mask,im2,seg2Mask,maxDelta,visualize)
    
    ind = find(seg1Mask);
    coord_rcz = Utils.IndToCoord(size(seg1Mask),ind);
    extents1_rcz = [min(coord_rcz,[],1);max(coord_rcz,[],1)];
    sz = extents1_rcz(2,:) - extents1_rcz(1,:) +1;
    size1 = min(sz(sz>1));
    
    ind = find(seg2Mask);
    coord_rcz = Utils.IndToCoord(size(seg2Mask),ind);
    extents2_rcz = [min(coord_rcz,[],1);max(coord_rcz,[],1)];
    sz = extents2_rcz(2,:) - extents2_rcz(1,:) +1;
    size2 = min(sz(sz>1));
        
    minOverlap = max(size1,size2);
    
    extents = [min(extents1_rcz(1,:),extents2_rcz(1,:));max(extents1_rcz(2,:),extents2_rcz(2,:))];
    if (size(extents,2)==2)
        extents(:,3) = [1;1];
    end
    extents(1,:) = max(extents(1,:)-10,ones(1,3));
    extents(2,:) = min(extents(2,:)+10,[size(im1,1),size(im1,2),size(im1,3)]);
    
    im1(~seg1Mask) = 0;
    im2(~seg2Mask) = 0;
    
    imROI1 = im1(extents(1,1):extents(2,1),extents(1,2):extents(2,2),extents(1,3):extents(2,3));
    imROI2 = im2(extents(1,1):extents(2,1),extents(1,2):extents(2,2),extents(1,3):extents(2,3));
  
    [deltaX,deltaY,deltaZ,maxNCV,volOverlap,ncvMatrixROI] = Registration.FFT.RegisterTwoImages(imROI1,[],imROI2,[],[],minOverlap,maxDelta,[],visualize);
    
    ncvMatrixROI = max(ncvMatrixROI,0);
    ncvMatrixROIsz = [size(ncvMatrixROI,1),size(ncvMatrixROI,2),size(ncvMatrixROI,3)];
    
    mid = size(ncvMatrixROI)/2;
    ncvExtents = [-floor(mid);floor(mid)+round(mod(mid,1))];
    mid = floor(mid);
    if (numel(mid)==2)
        ncvExtents(:,3) = [0;1];
        mid(3) = 1;
    end
    [X,Y,Z] = meshgrid(ncvExtents(1,2):ncvExtents(2,2)-1,ncvExtents(1,1):ncvExtents(2,1)-1,ncvExtents(1,3):ncvExtents(2,3)-1);
    
    cent = false(ncvMatrixROIsz);
    cent(mid(1),mid(2),mid(3)) = true;
    
    % make a mask that removes any out of range distances
    se = ImProc.MakeBallMask(maxDelta);
    midSE = ceil(size(se)/2);
    startSE = max(ones(1,3),midSE - mid +1);
    endSE = min(size(se),ncvMatrixROIsz + startSE -1);
    se = se(startSE(1):endSE(1),startSE(2):endSE(2),startSE(3):endSE(3));
    seSz = [size(se,1),size(se,2),size(se,3)];
    seFull = false(ncvMatrixROIsz);
    seStart = ceil((ncvMatrixROIsz - seSz)./2)+1;
    seStart = max(ones(1,numel(mid)),seStart);
    seEnd = seStart + seSz -1;
    seFull(seStart(1):seEnd(1),seStart(2):seEnd(2),seStart(3):seEnd(3)) = se;
    
    moveDist = double(bwdist(cent)); % distance transform
    moveDist = (max(moveDist(:)) - moveDist)./max(moveDist(:)); % flip the transform so that the center is maximal
    moveDist = moveDist ./ max(moveDist(:)); % normalize the center to 1
    moveDist(~seFull) = 0; % mask out the distances that are out of the bounds
    newNCV = ncvMatrixROI.*moveDist; % multiply the ncv by this new bias factor
    
    [v,I] = max(newNCV(:));
    coord_rcz = Utils.IndToCoord(size(newNCV),I);
    
    [cost,I] = max(newNCV(:));
    coord_rc = Utils.IndToCoord(size(newNCV),I);
    coord_xy = [X(1,coord_rc(2)),Y(coord_rc(1),1)];
    
    if (visualize)
        figure
        subplot(2,3,1)
        ImUtils.ThreeD.ShowMaxImage(imROI1,false,3,gca);
        title('T');
        
        subplot(2,3,4)
        ImUtils.ThreeD.ShowMaxImage(imROI2,false,3,gca);
        title('T+1');
        
        if (numel(coord_rcz)==3)
            z = min(coord_rcz(3));
        else
            z = 1;
        end
        
        subplot(2,3,[2,5])
        surf(X(:,:,z),Y(:,:,z),ncvMatrixROI(:,:,z),'LineStyle','none');
        colormap(gca,'parula');
        text(deltaX,deltaY,maxNCV,sprintf('(%d,%d,%d): %f',deltaX,deltaY,deltaZ,maxNCV));
        
        subplot(2,3,[3,6])
        surf(X(:,:,z),Y(:,:,z),newNCV(:,:,z,1),'LineStyle','none');
        colormap(gca,'parula');
        text(coord_xy(1),coord_xy(2),newNCV(I),sprintf('(%d,%d,%d): %f',coord_xy(1),coord_xy(2),deltaZ,cost));
    end
    
    cost = 1 - cost;
end
