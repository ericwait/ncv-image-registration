%CALCULATEOVERLAP returns ROIs in the form
%[columnStart,rowStart, zStart, columnEnd,rowEnd, zEnd]
function [image1ROI,image2ROI, minXdist, minYdist,currentPadding] = calculateOverlap(imageData1,imageData2,padding,unitFactor)

if (~exist('padding','var') || isempty(padding))
    padding = [0,0];
end

if (~exist('unitFactor','var') || isempty(unitFactor))
    unitFactor = 1e6;
end

currentPadding = padding;

image1ROI = [1, 1, 1, imageData1.XDimension, imageData1.YDimension, imageData1.ZDimension];
image2ROI = [1, 1, 1, imageData2.XDimension, imageData2.YDimension, imageData2.ZDimension];

image1Space = [ -imageData1.YPosition*unitFactor,...
                -imageData1.XPosition*unitFactor,...
                -imageData1.YPosition*unitFactor+imageData1.XPixelPhysicalSize*imageData1.XDimension,...
                -imageData1.XPosition*unitFactor+imageData1.YPixelPhysicalSize*imageData1.YDimension];

image2Space = [ -imageData2.YPosition*unitFactor,...
                -imageData2.XPosition*unitFactor,...
                -imageData2.YPosition*unitFactor+imageData2.XPixelPhysicalSize*imageData2.XDimension,...
                -imageData2.XPosition*unitFactor+imageData2.YPixelPhysicalSize*imageData2.YDimension];

if (image1Space(1)<image2Space(1))
    % image1 is left of image2
    image2ROI(1) = round((image2Space(1)-image1Space(1))/imageData2.XPixelPhysicalSize) +1;
    if (image2ROI(1)-padding(1)<1)
        currentPadding(1) = image2ROI(1) -1;
        image2ROI(1) = 1;
    else
        image2ROI(1) = image2ROI(1) - padding(1);
    end
    image1ROI(4) = image2ROI(4) - image2ROI(1) +1;
    minXdist = image1ROI(4);
else
    % image1 is right or alligned with image2
    image1ROI(1) = round((image1Space(1)-image2Space(1))/imageData1.XPixelPhysicalSize) +1;
    if (image1ROI(1)-padding(1)<1)
        currentPadding(1) = image1ROI(1) -1;
        image1ROI(1) = 1;
    else
        image1ROI(1) = image1ROI(1) - padding(1);
    end
    currentPadding(1) = -currentPadding(1);
    image2ROI(4) = image1ROI(4) - image1ROI(1) +1;
    minXdist = image2ROI(4);
end

if (image1Space(2)<image2Space(2))
    % image1 is above image2
    image2ROI(2) = round((image2Space(2)-image1Space(2))/imageData2.YPixelPhysicalSize) +1;
    if (image2ROI(2)-padding(2)<1)
        currentPadding(2) = image2ROI(2) -1;
        image2ROI(2) = 1;
    else
        image2ROI(2) = image2ROI(2) - padding(2);
    end
    image1ROI(5) = image2ROI(5) - image2ROI(2) +1;
    minYdist = image1ROI(5);
else
    % image1 is below image2
    image1ROI(2) = round((image1Space(2)-image2Space(2))/imageData1.YPixelPhysicalSize) +1;
    if (image1ROI(2)-padding(2)<1)
        currentPadding(2) = image1ROI(2) -1;
        image1ROI(2) = 1;
    else
        image1ROI(2) = image1ROI(2) - padding(2);
    end
    currentPadding(2) = -currentPadding(2);
    image2ROI(5) = image1ROI(5) - image1ROI(2) +1;
    minYdist = image2ROI(5);
end

image1ROI([1,2]) = max(image1ROI([1,2]),[1,1]);
image1ROI([4,5]) = min(image1ROI([4,5]),[imageData1.XDimension,imageData1.YDimension]);

image2ROI([1,2]) = max(image2ROI([1,2]),[1,1]);
image2ROI([4,5]) = min(image2ROI([4,5]),[imageData2.XDimension,imageData2.YDimension]);
end