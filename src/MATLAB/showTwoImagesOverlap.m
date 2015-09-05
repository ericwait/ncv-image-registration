function showTwoImagesOverlap(im1,im1Data,im2,im2Data,deltas)
%SHOWTWOIMAGESOVERLAP Summary of this function goes here
%   Detailed explanation goes here

if (~exist('deltas','var') || isempty(deltas))
    deltas = [0,0,0];
end

[im1ROI,im2ROI,~,~] = calculateOverlap(im1Data,im2Data);

im1Starts = [1,1,1];
im1Ends = [size(im1,2),size(im1,1),size(im1,3)];
im2Starts = [1,1,1];
im2Ends = [size(im2,2),size(im2,1),size(im2,3)];

for i=1:3
    if (im1ROI(i)==1 && im2ROI(i)==1)
        %aligned
        if (sign(deltas(i))>=0)
            %2nd needs to shift in the positive direction
            im1Starts(i) = 1;
            im2Starts(i) = deltas(i) +1;
        else
            %2nd needs to shift in the negative direction
            im1Starts(i) = abs(deltas(i)) +1;
            im2Starts(i) = 1;
        end
    elseif (im1ROI(i)>1)
        %2nd is in the positive direction
        if (sign(deltas(i))>=0)
            %2nd needs to shift in the positive direction
            im1Starts(i) = 1;
            im2Starts(i) = im1ROI(i) + deltas(i) +1;
        else
            %2nd needs to shift in the negative direction
            if (abs(deltas(i))>im1ROI(i))
                %the shift will put 2nd in the negative direction from 1st
                leftShift = im1ROI(i) + deltas(i);%becasue it is negative
                im1Starts(i) = leftShift +1;
                im2Starts(i) = 1;
            else
                %the shift will not move 2nd far enough to change sign
                im1Starts(i) = 1;
                im2Starts(i) = im1ROI(i) + deltas(i);
            end
        end
    else
        %2nd is in the negative direction
        if (sign(deltas(i))>=0)
            %2nd needs to shift in the positive direction
            if (deltas(i)>im2ROI(i))
                %the shift will put 2nd in the positive direction from 1st
                rightShift = im2ROI(i) - deltas(i);%becasue it is positive
                im1Starts(i) = 1;
                im2Starts(i) = rightShift +1;
            else
                %the shift will not move 2nd far enough to change sign
                im1Starts(i) = im2ROI(i) - deltas(i);
                im2Starts(i) = 1;
            end
        else
            %2nd needs to shift in the negative direction
            im1Starts(i) = im2ROI(i) - deltas(i);%because it is negative
            im2Starts(i) = 1;
        end
    end
    
    im1Ends(i) = im1Starts(i) + im1Ends(i)-1;
    im2Ends(i) = im2Starts(i) + im2Ends(i)-1;
end

combinedWidth = abs(im1ROI(1)-im2ROI(1)) + max(size(im1,2),size(im2,2));
combinedHeight = abs(im1ROI(2)-im2ROI(2)) + max(size(im1,1),size(im2,1));
combinedDepth = abs(im1ROI(3)-im2ROI(3)) + max(size(im1,3),size(im2,3));

im = zeros(combinedHeight,combinedWidth,3,combinedDepth,'like',im1);

im(im1Starts(2):im1Ends(2),im1Starts(1):im1Ends(1),1,im1Starts(3):im1Ends(3)) = im1;
im(im2Starts(2):im2Ends(2),im2Starts(1):im2Ends(1),3,im2Starts(3):im2Ends(3)) = im2;

figure
image(max(im,[],4));
axis image
end
