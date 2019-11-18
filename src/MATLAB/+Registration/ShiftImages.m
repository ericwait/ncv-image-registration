function newIm = ShiftImages(im1,im2,deltaX,deltaY,deltaZ,mode)
% Shift im2 realitive to im1. Positive deltas move the origin of im2 
% in the positive direction w.r.t. im1. A two channel registered image is 
% returned in which the first channel is im1 and the second channel is im2.
% There are three shift modes: pad (default), fit, and crop. Pad returns an image
% that is at least as large as the input images, and contains zero padding
% where the images do not overlap. Fit returns an image equal to the size
% of im1, and it maintains the reference coordinate system of im1. Crop
% returns an image at most as large as the input images, and contains only
% the intersection area.

    if (~exist('mode','var') || isempty(mode))
        mode = 'pad';
    end
    
    im1Sz = ImUtils.Size(im1);
    im2Sz = ImUtils.Size(im2);
    
    deltas = [deltaY,deltaX,deltaZ];
    
    posDeltas = max(deltas,[0,0,0]);
    negDeltas = min(deltas,[0,0,0]);
    
    refStart = [1,1,1] + posDeltas;
    refEnd = im1Sz(1:3) + negDeltas;
    trgtStart = [1,1,1] - negDeltas;
    trgtEnd = im2Sz(1:3) - posDeltas;
       
    if strcmp(mode,'pad')
        padSz = im1Sz(1:3) + abs(deltas);
        newIm = zeros([padSz,2,size(im1,5)],class(im1));

        refEnd = padSz + negDeltas;
        trgtEnd = padSz - posDeltas;
        
        newIm(trgtStart(1):trgtEnd(1),...
              trgtStart(2):trgtEnd(2),...
              trgtStart(3):trgtEnd(3),1,:) = im1;

        newIm(refStart(1):refEnd(1),...
              refStart(2):refEnd(2),...
              refStart(3):refEnd(3),2,:) = im2;
    elseif strcmp(mode,'fit')
        newIm = zeros([im1Sz(1),im1Sz(2),im1Sz(3),2,im1Sz(5)],class(im1));

        newIm(:,:,:,1,:) = im1;

        newIm(refStart(1):refEnd(1),...
              refStart(2):refEnd(2),...
              refStart(3):refEnd(3),2,:) = im2(trgtStart(1):trgtEnd(1),...
                                               trgtStart(2):trgtEnd(2),...
                                               trgtStart(3):trgtEnd(3));
    elseif strcmp(mode,'crop')
        cropSz = im1Sz(1:3) - abs(deltas);
        newIm = zeros([cropSz,2,im1Sz(5)],class(im1));

        newIm(:,:,:,1,:) = im1(refStart(1):refEnd(1),...
                               refStart(2):refEnd(2),...
                               refStart(3):refEnd(3));

        newIm(:,:,:,2,:) = im2(trgtStart(1):trgtEnd(1),...
                               trgtStart(2):trgtEnd(2),...
                               trgtStart(3):trgtEnd(3));
    else
        error('Not a valid shift mode');
    end
end
