%CALCULATEOVERLAP returns ROIs in the form
%[columnStart,rowStart, zStart, columnEnd,rowEnd, zEnd]
function [image1ROI,image2ROI, minXdist, minYdist] = calculateOverlap(imageData1,imageData2)

image1ROI = [1, 1, 1, imageData1.XDimension, imageData1.YDimension, imageData1.ZDimension];
image2ROI = [1, 1, 1, imageData2.XDimension, imageData2.YDimension, imageData2.ZDimension];

image1Space = [-imageData1.YPosition*1e6,...
    -imageData1.XPosition*1e6,...
    -imageData1.YPosition*1e6+imageData1.XPixelPhysicalSize*imageData1.XDimension,...
    -imageData1.XPosition*1e6+imageData1.YPixelPhysicalSize*imageData1.YDimension];

image2Space = [-imageData2.YPosition*1e6,...
    -imageData2.XPosition*1e6,...
    -imageData2.YPosition*1e6+imageData2.XPixelPhysicalSize*imageData2.XDimension,...
    -imageData2.XPosition*1e6+imageData2.YPixelPhysicalSize*imageData2.YDimension];

if (image1Space(1)<image2Space(1))
    image2ROI(1) = round((image2Space(1)-image1Space(1))/imageData2.XPixelPhysicalSize) +1;
    image1ROI(4) = image2ROI(4) - image2ROI(1) +1;
    minXdist = image2Space(1)-image1Space(3);
else
    image1ROI(1) = round((image1Space(1)-image2Space(1))/imageData1.XPixelPhysicalSize) +1;
    image2ROI(4) = image1ROI(4) - image1ROI(1) +1;
    minXdist = image1Space(1)-image2Space(3);
end

if (image1Space(2)<image2Space(2))
    image2ROI(2) = round((image2Space(2)-image1Space(2))/imageData2.YPixelPhysicalSize) +1;
    image1ROI(5) = image2ROI(5) - image2ROI(2) +1;
    minYdist = image2Space(2)-image1Space(4);
else
    image1ROI(2) = round((image1Space(2)-image2Space(2))/imageData1.YPixelPhysicalSize) +1;
    image2ROI(5) = image1ROI(5) - image1ROI(2) +1;
    minYdist = image1Space(2)-image2Space(4);
end

image1ROI([1,2]) = max(image1ROI([1,2]),[1,1]);
image1ROI([4,5]) = min(image1ROI([4,5]),[imageData1.XDimension,imageData1.YDimension]);

image2ROI([1,2]) = max(image2ROI([1,2]),[1,1]);
image2ROI([4,5]) = min(image2ROI([4,5]),[imageData2.XDimension,imageData2.YDimension]);
end