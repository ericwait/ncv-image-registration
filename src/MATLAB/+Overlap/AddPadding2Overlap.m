function [ image1ROIout,image2ROIout, varargout] = addPadding2Overlap(imageData1, image1ROI, imageData2, image2ROI, padding)
%ADDPADDING2OVERLAP Summary of this function goes here
%   Detailed explanation goes here

%% Ensure that there is padding for each dimension
if (length(padding)==1)
    padding = repmat(padding,1,3);
elseif (length(padding)==2)
    padding = [padding,0];
end

%% Create arrays that allow for 'for' loops
im1Dim = [imageData1.XDimension,imageData1.YDimension,imageData1.ZDimension];
im2Dim = [imageData2.XDimension,imageData2.YDimension,imageData2.ZDimension];

paddingUsed = zeros(1,3);
image1ROIout = image1ROI;
image2ROIout = image2ROI;

for curDim=1:3
    if (length(image1ROI(curDim):image1ROI(curDim+3))==im1Dim(curDim) || ...
        length(image2ROI(curDim):image2ROI(curDim+3))==im2Dim(curDim))
        % This means that there is no more room to grow in this dimension
            continue
    end
    
    if (image1ROI(curDim)~=1 && image2ROI(curDim)==1)
        %This means that image1 is before image2 in this
        %dimension (start pos of image1 is smaller than image2)
        
        % Find the max padding possible
        paddingUsed(curDim) = min(padding(curDim),image1ROI(curDim));
        paddingUsed(curDim) = min(paddingUsed(curDim),im2Dim(curDim)-image2ROI(curDim+3)+1);
        
        % Apply the padding
        image1ROIout(curDim) = image1ROIout(curDim) - paddingUsed(curDim);
        image2ROIout(curDim+3) = image2ROIout(curDim+3) + paddingUsed(curDim);
        
        %this is like having the images shift closer together
        paddingUsed(curDim) = -paddingUsed(curDim);
    elseif (image1ROI(curDim)==1 && image2ROI(curDim)==1)
        %This means that the start of each image is alligned
        
        % Find the max padding possible
        paddingUsed(curDim) = min(padding(curDim),im1Dim(curDim)-image1ROI(curDim+3)+1);
        paddingUsed(curDim) = min(paddingUsed(curDim),im2Dim(curDim)-image2ROI(curDim+3)+1);
        
        % Apply the padding
        image1ROIout(curDim+3) = image1ROIout(curDim+3) + paddingUsed(curDim);
        image2ROIout(curDim+3) = image2ROIout(curDim+3) + paddingUsed(curDim);
    else
        %This means that image2 is before image1 in this
        %dimension (start pos of image2 is smaller than image1)
        
        % Find the max padding possible
        paddingUsed(curDim) = min(padding(curDim),image2ROI(curDim));
        paddingUsed(curDim) = min(paddingUsed(curDim),im1Dim(curDim)-image1ROI(curDim+3)+1);
        
        % Apply the padding
        image2ROIout(curDim) = image2ROIout(curDim) - paddingUsed(curDim);
        image1ROIout(curDim+3) = image1ROIout(curDim+3) + paddingUsed(curDim);
    end
end

%% Pass optional param out 
if (nargout>2)
    varargout{1} = paddingUsed;
end
end

