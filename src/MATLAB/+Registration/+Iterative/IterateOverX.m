function normCoSquare = IterateOverX(maxIterX,maxIterY,im1,im2,xStart1,xStart2,yStart1,yStart2,curDeltaZ,minOverlap,visualize)
global Rect1 Rect2
normCoSquare = zeros(maxIterY*2,maxIterX*2);

for delta = 1:maxIterX*2
    curDelta = delta-maxIterX;
    [start1,start2,end1,end2] = Registration.Overlap.CalculateROIs(curDelta,xStart1,xStart2,size(im1,2),size(im2,2));
    if (end1-start1<minOverlap || end2-start2<minOverlap), continue, end
    
    if (visualize==1)
        pos1 = get(Rect1,'Position');
        pos2 = get(Rect2,'Position');
        set(Rect1,'Position',[max(start1,1),max(pos1(2),1),max(end1-start1,1),max(pos1(4),1)]);
        set(Rect2,'Position',[max(start2,1),max(pos2(2),1),max(end2-start2,1),max(pos2(4),1)]);
    end
    
    imX1 = im1(:,start1:end1,:);
    imX2 = im2(:,start2:end2,:);
    normCoSquare(:,delta) = Registration.Iterative.IterateOverY(maxIterY,imX1,imX2,curDelta,yStart1,yStart2,curDeltaZ,minOverlap,visualize);
end
end
