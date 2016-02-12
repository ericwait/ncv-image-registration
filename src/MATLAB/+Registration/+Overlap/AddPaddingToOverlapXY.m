function [ image1ROIout_XY,image2ROIout_XY, varargout] = AddPaddingToOverlapXY(imageData1, image1ROI_XY, imageData2, image2ROI_XY, padding)
%ADDPADDINGTOOVERLAP Summary of this function goes here
%   Detailed explanation goes here

%% Ensure that there is padding for each dimension
if (length(padding)==1)
    padding = repmat(padding,1,3);
elseif (length(padding)==2)
    padding = [padding,0];
end

%% Create arrays that allow for 'for' loops
im1Dim_XY = imageData1.Dimensions;
im2Dim = imageData2.Dimensions;

paddingUsed_XY = zeros(1,3);
image1ROIout_XY = image1ROI_XY;
image2ROIout_XY = image2ROI_XY;

%% Calculate the ROIs
for curDim=1:3
    if (length(image1ROI_XY(curDim):image1ROI_XY(curDim+3))==im1Dim_XY(curDim) || ...
        length(image2ROI_XY(curDim):image2ROI_XY(curDim+3))==im2Dim(curDim))
        % This means that there is no more room to grow in this dimension
            continue
    end
    
    if (image1ROI_XY(curDim)~=1 && image2ROI_XY(curDim)==1)
        %This means that image1 is before image2 in this
        %dimension (start pos of image1 is smaller than image2)
        
        % Find the max padding possible
        paddingUsed_XY(curDim) = min(padding(curDim),image1ROI_XY(curDim));
        paddingUsed_XY(curDim) = min(paddingUsed_XY(curDim),im2Dim(curDim)-image2ROI_XY(curDim+3)+1);
        paddingUsed_XY(curDim) = paddingUsed_XY(curDim) -1;
        
        % Apply the padding
        image1ROIout_XY(curDim) = image1ROIout_XY(curDim) - paddingUsed_XY(curDim);
        image2ROIout_XY(curDim+3) = image2ROIout_XY(curDim+3) + paddingUsed_XY(curDim);
        
        %this is like having the images shift closer together
        paddingUsed_XY(curDim) = -paddingUsed_XY(curDim);
    elseif (image1ROI_XY(curDim)==1 && image2ROI_XY(curDim)==1)
        %This means that the start of each image is alligned
        
        % Find the max padding possible
        paddingUsed_XY(curDim) = min(padding(curDim),im1Dim_XY(curDim)-image1ROI_XY(curDim+3)+1);
        paddingUsed_XY(curDim) = min(paddingUsed_XY(curDim),im2Dim(curDim)-image2ROI_XY(curDim+3)+1);
        paddingUsed_XY(curDim) = paddingUsed_XY(curDim) -1;
        
        % Apply the padding
        image1ROIout_XY(curDim+3) = image1ROIout_XY(curDim+3) + paddingUsed_XY(curDim);
        image2ROIout_XY(curDim+3) = image2ROIout_XY(curDim+3) + paddingUsed_XY(curDim);
    else
        %This means that image2 is before image1 in this
        %dimension (start pos of image2 is smaller than image1)
        
        % Find the max padding possible
        paddingUsed_XY(curDim) = min(padding(curDim),image2ROI_XY(curDim));
        paddingUsed_XY(curDim) = min(paddingUsed_XY(curDim),im1Dim_XY(curDim)-image1ROI_XY(curDim+3)+1);
        paddingUsed_XY(curDim) = paddingUsed_XY(curDim) -1;
        
        % Apply the padding
        image2ROIout_XY(curDim) = image2ROIout_XY(curDim) - paddingUsed_XY(curDim);
        image1ROIout_XY(curDim+3) = image1ROIout_XY(curDim+3) + paddingUsed_XY(curDim);
    end
end

%% Pass optional param out 
if (nargout>2)
    varargout{1} = paddingUsed_XY;
end
end

