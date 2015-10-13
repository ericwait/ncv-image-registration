function normCoLine = IterateOverY(maxIterY,im1,im2,curDeltaX,yStart1,yStart2,curDeltaZ,minOverlap,visualize)
global Rect1 Rect2
normCoLine = zeros(maxIterY*2,1);

for delta = 1:maxIterY*2
    curDelta = delta-maxIterY;
    [start1,start2,end1,end2] = Registration.Overlap.CalculateROIs(curDelta,yStart1,yStart2,size(im1,1),size(im2,1));
    if (end1-start1<minOverlap || end2-start2<minOverlap), continue, end
    
    imY1 = im1(start1:end1,:,:);
    imY2 = im2(start2:end2,:,:);
    
    normCoLine(delta) = Math.NormalizedCovariance(imY1, imY2);
    
    if (visualize==1)
        pos1 = get(Rect1,'Position');
        pos2 = get(Rect2,'Position');
        set(Rect1,'Position',[max(pos1(1),1),max(start1,1),max(pos1(3),1),max(end1-start1,1)]);
        set(Rect2,'Position',[max(pos2(1),1),max(start2,1),max(pos2(3),1),max(end2-start2,1)]);
        Registration.Iterative.UpdateXYviewer(imY1,imY2,normCoLine(delta),curDeltaX,curDelta,curDeltaZ);
    end
    
    if (normCoLine(delta)>1 || normCoLine(delta)<-1)
        warning('Recived a NCV out of bounds:%f, overlap:(%d,%d,%d)',normCoLine(delta),size(imY1,2),size(imY1,1),size(imY1,3));
        normCoLine(delta) = 0;
    end
end
end
