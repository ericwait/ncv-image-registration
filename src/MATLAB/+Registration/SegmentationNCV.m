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
    extents(1,:) = max(extents(1,:)-10,ones(1,3));
    extents(2,:) = min(extents(2,:)+10,[size(im1,1),size(im1,2),size(im1,3)]);
    
    im1(~seg1Mask) = 0;
    im2(~seg2Mask) = 0;
    
    imROI1 = im1(extents(1,1):extents(2,1),extents(1,2):extents(2,2),extents(1,3):extents(2,3));
    imROI2 = im2(extents(1,1):extents(2,1),extents(1,2):extents(2,2),extents(1,3):extents(2,3));
  
    [deltaX,deltaY,deltaZ,maxNCV,volOverlap,ncvMatrixROI] = Registration.FFT.RegisterTwoImages(imROI1,[],imROI2,[],[],minOverlap,maxDelta,[],visualize);
    
    ncvMatrixROI = max(ncvMatrixROI,0);
    
    mid = ceil(size(ncvMatrixROI)/2);
    ncvExtents = [-mid+1;mid];
    if (numel(mid)==2)
        ncvExtents(:,3) = 0;
    end
    [X,Y,Z] = meshgrid(ncvExtents(1,2):ncvExtents(2,2)-1,ncvExtents(1,1):ncvExtents(2,1)-1,ncvExtents(1,3):ncvExtents(2,3)-1);
    
    cent = false(size(ncvMatrixROI));
    cent(mid(1),mid(2),mid(3)) = true;
    moveDist = double(bwdist(cent));
    moveDist = moveDist./maxDelta;
    newNCV = ncvMatrixROI-moveDist;
    newNCV(newNCV<0) = 0;
    
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
        ImUtils.ThreeD.ShowMaxImage(imROI1,false,3,gca);
        title('T+1');
        
        subplot(2,3,[2,5])
        surf(X(:,:,min(coord_rcz(3),1)),Y(:,:,min(coord_rcz(3)),1),ncvMatrixROI(:,:,min(coord_rcz(3),1)),'LineStyle','none');
        colormap(gca,'parula');
        text(deltaX,deltaY,maxNCV,sprintf('(%d,%d,%d): %f',deltaX,deltaY,deltaZ,maxNCV));
        
        subplot(2,3,[3,6])
        surf(X,Y,newNCV(:,:,min(coord_rcz(3),1)),'LineStyle','none');
        colormap(gca,'parula');
        text(coord_xy(1),coord_xy(2),newNCV(I),sprintf('(%d,%d,%d): %f',coord_xy(1),coord_xy(2),deltaZ,cost));
    end
    
    cost = 1 - cost;
end
