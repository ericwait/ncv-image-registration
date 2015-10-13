function normCoSquare = IterateOverXpar(maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,curDeltaZ,minOverlap)
normCoSquare = zeros(maxIterY*2,maxIterX*2);

parfor delta = 1:maxIterX*2
    curDelta = delta-maxIterX;
    [start1,start2,end1,end2] = Registration.Overlap.CalculateROIs(curDelta,xStart1,xStart2,size(im1,2),size(im2,2));
    if (end1-start1<minOverlap || end2-start2<minOverlap), continue, end
    
    imX1 = im1(:,start1:end1,:);
    imX2 = im2(:,start2:end2,:);
    normCoSquare(:,delta) = Registration.Iterative.IterateOverY(maxIterY,imX1,imX2,curDelta,yStart1,yStart2,curDeltaZ,minOverlap,false);
end
end
