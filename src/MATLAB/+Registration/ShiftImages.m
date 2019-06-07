function newIm = ShiftImages(im1,im2,deltaX,deltaY,deltaZ)
% Shift image 2 realitive to image 1. Positive delta moves the origin 
%   of image 2 further into image 1.
% Image 1 will be in the first channel and Image 2 will be in the second.

    deltas_rc = [deltaY,deltaX,deltaZ];
    
    im1Sz = ImUtils.Size(im1);
    im2Sz = ImUtils.Size(im2);

    im1Start_rc = max(-deltas_rc+1,[1,1,1]);
    im2Start_rc = max(deltas_rc+1,[1,1,1]);
    
    im1End_rc = im1Start_rc +im1Sz(1:3) -1;
    im2End_rc = im2Start_rc +im2Sz(1:3) -1;
    
    newSize = max(im1End_rc,im2End_rc);
    newIm = zeros([newSize,2,size(im1,5)],class(im1));
    
    newIm(im1Start_rc(1):im1End_rc(1),...
          im1Start_rc(2):im1End_rc(2),...
          im1Start_rc(3):im1End_rc(3),1,:) = im1;
      
    newIm(im2Start_rc(1):im2End_rc(1),...
          im2Start_rc(2):im2End_rc(2),...
          im2Start_rc(3):im2End_rc(3),2,:) = im2;        
end
