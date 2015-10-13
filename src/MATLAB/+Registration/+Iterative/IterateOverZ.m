function normCoCube = IterateOverZ(maxIterZ,maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,zStart1,zStart2,minOverlap,visualize)
normCoCube = zeros(maxIterY*2,maxIterX*2,maxIterZ*2);

for delta = 1:maxIterZ*2
    curDelta = delta-maxIterZ;
    [start1,start2,end1,end2] = Registration.Overlap.CalculateROIs(curDelta,zStart1,zStart2,size(im1,3),size(im2,3));
    if (end1-start1<minOverlap/5 || end2-start2<minOverlap/5), continue, end
    
    imZ1 = im1(:,:,start1:end1);
    imZ2 = im2(:,:,start2:end2);
    normCoCube(:,:,delta) = Registration.Iterative.IterateOverX(maxIterX,maxIterY,imZ1,imZ2,xStart1,xStart2,...
        yStart1,yStart2,curDelta,minOverlap,false);
end
end
