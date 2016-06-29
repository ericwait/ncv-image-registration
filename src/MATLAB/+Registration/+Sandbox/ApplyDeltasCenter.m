function [im1Out,im1BWout,im2Out,im2BWout] = ApplyDeltasCenter(im1,im2,deltas_rc)
if (numel(deltas_rc>3))
    im2BW = true(size(im2));
    im2 = imrotate(im2,deltas_rc(4),'bicubic');
    im2BW = imrotate(im2BW,deltas_rc(4),'bicubic');
end

sz1 = [size(im1,1),size(im1,2),size(im1,3),size(im1,4),size(im1,5)];
sz2 = [size(im2,1),size(im2,2),size(im2,3),size(im2,4),size(im2,5)];

halfSz1 = sz1./2;
halfSz2 = sz2./2;

im1BW = true(size(im1));

im1Start_rc = ones(1,3);
im1Ends_rc = zeros(1,3);
im2Start_rc = ones(1,3);
im2Ends_rc = zeros(1,3);
for d=1:3
    intCenter = max(halfSz1(d),halfSz2(d));
    im1center = intCenter;
    im2center = intCenter + deltas_rc(d);

    start1 = floor(im1center - halfSz1(d));
    start2 = floor(im2center - halfSz2(d));
    
    if (start1<start2)
        % The origins of one is before two
        im1Start_rc(d) = 1;
        im2Start_rc(d) = start2 + abs(start1) -1;
    elseif (start1>start2)
        % The origins of two is before one
        im1Start_rc(d) = start1 + abs(start2) -1;
        im2Start_rc(d) = 1;
    else
        % The origins are aligned
    end
    
    im1Ends_rc(d) = im1Start_rc(d) + sz1(d) -1;
    im2Ends_rc(d) = im2Start_rc(d) + sz2(d) -1;
end

im1Out = zeros(im1Ends_rc(1),im1Ends_rc(2),im1Ends_rc(3),sz1(4),sz1(5),'like',im1);
im2Out = zeros(im2Ends_rc(1),im2Ends_rc(2),im2Ends_rc(3),sz2(4),sz2(5),'like',im2);

im1BWout = false(im1Ends_rc(1),im1Ends_rc(2),im1Ends_rc(3),size(im1,4),size(im1,5));
im2BWout = false(im2Ends_rc(1),im2Ends_rc(2),im2Ends_rc(3),size(im2,4),size(im2,5));

im1Out(im1Start_rc(1):im1Start_rc(1)+sz1(1)-1,im1Start_rc(2):im1Start_rc(2)+sz1(2)-1,im1Start_rc(3):im1Start_rc(3)+sz1(3)-1,:,:) = im1;
im2Out(im2Start_rc(1):im2Start_rc(1)+sz2(1)-1,im2Start_rc(2):im2Start_rc(2)+sz2(2)-1,im2Start_rc(3):im2Start_rc(3)+sz2(3)-1,:,:) = im2;

im1BWout(im1Start_rc(1):im1Start_rc(1)+sz1(1)-1,im1Start_rc(2):im1Start_rc(2)+sz1(2)-1,im1Start_rc(3):im1Start_rc(3)+sz1(3)-1,:,:) = im1BW;
im2BWout(im2Start_rc(1):im2Start_rc(1)+sz2(1)-1,im2Start_rc(2):im2Start_rc(2)+sz2(2)-1,im2Start_rc(3):im2Start_rc(3)+sz2(3)-1,:,:) = im2BW;
end