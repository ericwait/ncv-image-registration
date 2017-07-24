function [deltaX,deltaY,deltaZ,maxNCV,volOverlap] = SegmentationNCV(im1,seg1Mask,im2,seg2Mask,maxDelta)
    [r,c,z] = find(seg1Mask);
    extents1 = [[min(r),min(c),min(z)];[max(r),max(c),max(z)]];
    size1 = min(extents1(2,:) - extents1(1,:));
    
    [r,c,z] = find(seg2Mask);
    extents2 = [[min(r),min(c),min(z)];[max(r),max(c),max(z)]];
    size2 = min(extents2(2,:) - extents2(1,:));
    
    minOverlap = max(size1,size2);
    
    extents = [min(extents1(1,:),extents2(1,:));max(extents1(2,:),extents2(2,:))];
    extents(1,:) = max(extents(1,:)-10,ones(1,3));
    extents(2,:) = min(extents(2,:)+10,[size(im1,1),size(im1,2),size(im1,3)]);
    
    im1(~seg1Mask) = 0;
    im2(~seg2Mask) = 0;
    
    imROI1 = im1(extents(1,1):extents(2,1),extents(1,2):extents(2,2),extents(1,3):extents(2,3));
    imROI2 = im2(extents(1,1):extents(2,1),extents(1,2):extents(2,2),extents(1,3):extents(2,3));
  
    [deltaX,deltaY,deltaZ,maxNCV,volOverlap] = Registration.FFT.RegisterTwoImages(imROI1,[],imROI2,[],[],minOverlap,maxDelta);
end
