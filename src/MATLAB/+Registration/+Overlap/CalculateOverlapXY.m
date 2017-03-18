function [image1ROI,image2ROI, minXdist, minYdist] = CalculateOverlapXY(imageData1,imageData2,unitFactor)
%CALCULATEOVERLAP returns ROIs in the form
%  [columnStart, rowStart, zStart, columnEnd, rowEnd, zEnd]
%  The UNIT_FACTOR changes the stage position into the same scale as the
%  voxel size

if (~exist('unitFactor','var') || isempty(unitFactor))
    unitFactor = 1e6;
end
if (~isfield(imageData1,'Position'))
    imageData1.Position = [0,0,0];
end
if (~isfield(imageData2,'Position'))
    imageData2.Position = [0,0,0];
end
if (~isfield(imageData1,'PixelPhysicalSize'))
    imageData1.PixelPhysicalSize = [1,1,1];
end
if (~isfield(imageData2,'PixelPhysicalSize'))
    imageData2.PixelPhysicalSize = [1,1,1];
end

%% default values
image1ROI = [1, 1, 1, imageData1.Dimensions];
image2ROI = [1, 1, 1, imageData2.Dimensions];

minXdist = inf;
minYdist = inf;

%% check if the data is consistant
if (any(imageData1.PixelPhysicalSize ~= imageData2.PixelPhysicalSize))
        warning('Images are not in the same physical space!');
        return
end

% the unit factor changes the stage position into the same scale as the
% voxel size
im1Pos = imageData1.Position * unitFactor;
im2Pos = imageData2.Position * unitFactor;

% only need one of these because of the previous check
phyVoxSize = imageData1.PixelPhysicalSize;
            
%% Find the overlapping regions
posDif = (im1Pos - im2Pos) ./ phyVoxSize;
posDif = round(posDif); %because this needs to be in voxels

minDis = ones(1,3) * inf;

for curDim=1:length(posDif)
    if (posDif(curDim) < 0)
        %this means that image1 is before image2 in this
        %dimension (start pos of image1 is smaller than image2)
        if (abs(posDif(curDim)) > imageData1.Dimensions(curDim))
            %the start position of image2 falls outside image1 (does not
            %overlap)
            image1ROI(curDim) = 0;
            image2ROI(curDim) = 0;
            image1ROI(curDim+3) = 0;
            image2ROI(curDim+3) = 0;
            minDis(curDim) = abs(posDif(curDim)) - imageData1.Dimensions(curDim);
        else
            %the start position of image2 falls w/in image1
            image1ROI(curDim) = abs(posDif(curDim));
            image2ROI(curDim) = 1;
            
            %set the ending points
            if (imageData1.Dimensions(curDim)-image1ROI(curDim) < imageData2.Dimensions(curDim))
                %this means that image2's end falls outside of image1's end
                image1ROI(curDim+3) = imageData1.Dimensions(curDim);
                image2ROI(curDim+3) = imageData2.Dimensions(curDim) - abs(posDif(curDim)) +1;
            elseif (imageData1.Dimensions(curDim)-image1ROI(curDim) == imageData2.Dimensions(curDim))
                %this means that image2's end falls exactly were image1's
                %end does
                image1ROI(curDim+3) = imageData1.Dimensions(curDim);
                image2ROI(curDim+3) = imageData2.Dimensions(curDim);
            else
                %this means that image2 exists w/in image1
                image1ROI(curDim+3) = image1ROI(curDim) + imageData2.Dimensions(curDim);
                image2ROI(curDim+3) = imageData2.Dimensions(curDim);
            end
            
            minDis(curDim) = 0;
        end
    elseif (posDif(curDim) == 0)
        %this means that the start of each image is alligned
        image1ROI(curDim) = 1;
        image2ROI(curDim) = 1;
        
        endPt = min(imageData1.Dimensions(curDim),imageData2.Dimensions(curDim));
        image1ROI(curDim+3) = endPt;
        image2ROI(curDim+3) = endPt;
        
        minDis(curDim) = 0;
    else
        %this means that image2 is before image1 in this
        %dimension (start pos of image2 is smaller than image1)
        if (posDif(curDim) > imageData2.Dimensions(curDim))
            %the start position of image1 falls outside image2 (does not
            %overlap)
            image1ROI(curDim) = 0;
            image2ROI(curDim) = 0;
            image1ROI(curDim+3) = 0;
            image2ROI(curDim+3) = 0;
            minDis(curDim) = posDif(curDim) - imageData2.Dimensions(curDim);
        else
            %the start position of image1 falls w/in image2
            image2ROI(curDim) = posDif(curDim);
            image1ROI(curDim) = 1;
            
            %set the ending points
            if (imageData2.Dimensions(curDim)-image2ROI(curDim) <= imageData1.Dimensions(curDim))
                %this means that image1's end falls outside of image2's end
                image2ROI(curDim+3) = imageData2.Dimensions(curDim);
                image1ROI(curDim+3) = imageData1.Dimensions(curDim) - posDif(curDim) +1;
            elseif (imageData2.Dimensions(curDim)-image2ROI(curDim) == imageData1.Dimensions(curDim))
                %this means that image1's end falls exactly were image2's
                %end does
                image1ROI(curDim+3) = imageData1.Dimensions(curDim);
                image2ROI(curDim+3) = imageData2.Dimensions(curDim);
            else
                %this means that image1 exists w/in image2
                image2ROI(curDim+3) = image2ROI(curDim) + imageData1.Dimensions(curDim);
                image1ROI(curDim+3) = imageData1.Dimensions(curDim);
            end            
            
            minDis(curDim) = 0;
        end
    end
end

%% return the min dist between the two images
minXdist = minDis(1);
minYdist = minDis(2);
end