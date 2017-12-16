function cost = SegmentationNCV(im1,seg1Mask,im2,seg2Mask,maxDelta,visualize)
    [r,c,z] = find(seg1Mask);
    extents1 = [[min(r),min(c),min(z)];[max(r),max(c),max(z)]];
    sz = extents1(2,:) - extents1(1,:) +1;
    size1 = min(sz(sz>1));
    
    [r,c,z] = find(seg2Mask);
    extents2 = [[min(r),min(c),min(z)];[max(r),max(c),max(z)]];
    sz = extents2(2,:) - extents2(1,:) +1;
    size2 = min(sz(sz>1));
    
    minOverlap = max(size1,size2);
    
    extents = [min(extents1(1,:),extents2(1,:));max(extents1(2,:),extents2(2,:))];
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
    [X,Y,Z] = meshgrid(ncvExtents(1,2):ncvExtents(2,2),ncvExtents(1,1):ncvExtents(2,1),ncvExtents(1,3):ncvExtents(2,3));
    
    cent = false(size(ncvMatrixROI));
    cent(mid(1),mid(2)) = true;
    moveDist = double(bwdist(cent));
    moveDist = moveDist./maxDelta;
    new = ncvMatrixROI-moveDist;
    new(new<0) = 0;
    
    [cost,I] = max(new(:));
    coord_rc = Utils.IndToCoord(size(new),I);
    coord_xy = [X(1,coord_rc(2)),Y(coord_rc(1),1)];
    
    if (visualize)
        figure
        subplot(2,3,1)
        imagesc(imROI1);
        colormap(gca,'gray');
        axis image
        title('T');
        
        subplot(2,3,4)
        imagesc(imROI2);
        colormap(gca,'gray');
        axis image
        title('T+1');
        
        subplot(2,3,[2,5])
        surf(X,Y,ncvMatrixROI,'LineStyle','none');
        colormap(gca,'parula');
        text(deltaX,deltaY,maxNCV,sprintf('(%d,%d,%d): %f',deltaX,deltaY,deltaZ,maxNCV));
        
        subplot(2,3,[3,6])
        surf(X,Y,new,'LineStyle','none');
        colormap(gca,'parula');
        text(coord_xy(1),coord_xy(2),new(I),sprintf('(%d,%d,%d): %f',coord_xy(1),coord_xy(2),deltaZ,cost));
    end
    
    cost = 1 - cost;
end
