function [image1ROI,image2ROI, minXdist, minYdist] = CalculateOverlapXY(imageData1,imageData2,unitFactor)
%CALCULATEOVERLAP returns ROIs in the form
%  [columnStart, rowStart, zStart, columnEnd, rowEnd, zEnd]
%  The UNIT_FACTOR changes the stage position into the same scale as the
%  voxel size

if (~exist('unitFactor','var') || isempty(unitFactor))
    unitFactor = 1e6;
end
if (~isfield(imageData1,'ZPosition'))
    imageData1.ZPosition = 0;
end
if (~isfield(imageData2,'ZPosition'))
    imageData2.ZPosition = 0;
end
if (~isfield(imageData1,'ZPixelPhysicalSize'))
    imageData1.ZPixelPhysicalSize = 1;
end
if (~isfield(imageData2,'ZPixelPhysicalSize'))
    imageData2.ZPixelPhysicalSize = 1;
end

%% default values
image1ROI = [1, 1, 1, imageData1.XDimension, imageData1.YDimension, imageData1.ZDimension];
image2ROI = [1, 1, 1, imageData2.XDimension, imageData2.YDimension, imageData2.ZDimension];

minXdist = inf;
minYdist = inf;

%% check if the data is consistant
if (imageData1.XPixelPhysicalSize ~= imageData2.XPixelPhysicalSize || ...
    imageData1.YPixelPhysicalSize ~= imageData2.YPixelPhysicalSize || ...
    imageData1.ZPixelPhysicalSize ~= imageData2.ZPixelPhysicalSize)
        warning('Images are not in the same physical space!');
        return
end

%% create arrays that allow for 'for' loops
im1Dim = [imageData1.XDimension,imageData1.YDimension,imageData1.ZDimension];
im2Dim = [imageData2.XDimension,imageData2.YDimension,imageData2.ZDimension];

% the unit factor changes the stage position into the same scale as the
% voxel size
im1Pos = [imageData1.XPosition,imageData1.YPosition,imageData1.ZPosition] * unitFactor;
im2Pos = [imageData2.XPosition,imageData2.YPosition,imageData2.ZPosition] * unitFactor;

% only need one of these because of the previous check
phyVoxSize = [imageData1.XPixelPhysicalSize,imageData1.YPixelPhysicalSize,imageData1.ZPixelPhysicalSize];
            
%% Find the overlapping regions
posDif = (im1Pos - im2Pos) .* phyVoxSize;
posDif = round(posDif); %because this needs to be in voxels

minDis = ones(1,3) * inf;

for curDim=1:length(posDif)
    if (posDif(curDim) < 0)
        %this means that image1 is before image2 in this
        %dimension (start pos of image1 is smaller than image2)
        if (abs(posDif(curDim)) > im1Dim(curDim))
            %the start position of image2 falls outside image1 (does not
            %overlap)
            image1ROI(curDim) = 0;
            image2ROI(curDim) = 0;
            image1ROI(curDim+3) = 0;
            image2ROI(curDim+3) = 0;
            minDis(curDim) = abs(posDif(curDim)) - im1Dim(curDim);
        else
            %the start position of image2 falls w/in image1
            image1ROI(curDim) = abs(posDif(curDim));
            image2ROI(curDim) = 1;
            
            %set the ending points
            if (im1Dim(curDim)-image1ROI(curDim) < im2Dim(curDim))
                %this means that image2's end falls outside of image1's end
                image1ROI(curDim+3) = im1Dim(curDim);
                image2ROI(curDim+3) = im2Dim(curDim) - abs(posDif(curDim)) +1;
            elseif (im1Dim(curDim)-image1ROI(curDim) == im2Dim(curDim))
                %this means that image2's end falls exactly were image1's
                %end does
                image1ROI(curDim+3) = im1Dim(curDim);
                image2ROI(curDim+3) = im2Dim(curDim);
            else
                %this means that image2 exists w/in image1
                image1ROI(curDim+3) = image1ROI(curDim) + im2Dim(curDim);
                image2ROI(curDim+3) = im2Dim(curDim);
            end
            
            minDis(curDim) = 0;
        end
    elseif (posDif(curDim) == 0)
        %this means that the start of each image is alligned
        image1ROI(curDim) = 1;
        image2ROI(curDim) = 1;
        
        endPt = min(im1Dim(curDim),im2Dim(curDim));
        image1ROI(curDim+3) = endPt;
        image2ROI(curDim+3) = endPt;
        
        minDis(curDim) = 0;
    else
        %this means that image2 is before image1 in this
        %dimension (start pos of image2 is smaller than image1)
        if (posDif(curDim) > im2Dim(curDim))
            %the start position of image1 falls outside image2 (does not
            %overlap)
            image1ROI(curDim) = 0;
            image2ROI(curDim) = 0;
            image1ROI(curDim+3) = 0;
            image2ROI(curDim+3) = 0;
            minDis(curDim) = posDif(curDim) - im2Dim(curDim);
        else
            %the start position of image1 falls w/in image2
            image2ROI(curDim) = posDif(curDim);
            image1ROI(curDim) = 1;
            
            %set the ending points
            if (im2Dim(curDim)-image2ROI(curDim) <= im1Dim(curDim))
                %this means that image1's end falls outside of image2's end
                image2ROI(curDim+3) = im2Dim(curDim);
                image1ROI(curDim+3) = im1Dim(curDim) - posDif(curDim) +1;
            elseif (im2Dim(curDim)-image2ROI(curDim) == im1Dim(curDim))
                %this means that image1's end falls exactly were image2's
                %end does
                image1ROI(curDim+3) = im1Dim(curDim);
                image2ROI(curDim+3) = im2Dim(curDim);
            else
                %this means that image1 exists w/in image2
                image2ROI(curDim+3) = image2ROI(curDim) + im1Dim(curDim);
                image1ROI(curDim+3) = im1Dim(curDim);
            end            
            
            minDis(curDim) = 0;
        end
    end
end

%% return the min dist between the two images
minXdist = minDis(1);
minYdist = minDis(2);
end