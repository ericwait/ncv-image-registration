function [deltas_RC,maxNCV,ncvMatrixROI] = GetMaxNCVdeltas(im1,im2,minOverlapVolume,maxSearchSize,orginCoords_RC,showDecisionSurf,chan,ang, im1Mask, im2Mask,saveFrames )
%[deltas,maxNCV] = RegisterTwoImages(im1,im2,minOverlapVolume)
% DELTAS is the shift of the upper left coners ....

if (~exist('orginCoords_RC','var') || isempty(orginCoords_RC))
    orginCoords_RC = zeros(1,ndims(im2));
end
if (~exist('maxSearchSize','var') || isempty(maxSearchSize))
    maxSearchSize = max(max(size(im1),size(im2)));
end
if (~exist('showDecisionSurf','var') || isempty(showDecisionSurf))
    showDecisionSurf = false;
end
if (~exist('chan','var') || isempty(chan))
    chan = 0;
end
if (~exist('ang','var') || isempty(ang))
    ang = 0;
end
if (~exist('im1Mask','var'))
    im1Mask = [];
end
if (~exist('im2Mask','var'))
    im2Mask = [];
end
if (~exist('saveFrames','var') || isempty(saveFrames))
    saveFrames = false;
end

ncvMatrix = Registration.FFT.NormalizedCovariance(im1,im2,minOverlapVolume,im1Mask,im2Mask);

%% cut out the region of interest
dims = ndims( ncvMatrix );
subsCell2 = cell(1, dims );

cntrPt_RC = size(im2) - orginCoords_RC;
searchBounds_RC =  round([max(cntrPt_RC - maxSearchSize, ones(1,ndims(im2)));...
    min(cntrPt_RC + maxSearchSize, size(ncvMatrix))]);

fullOrgin_RC = cntrPt_RC - searchBounds_RC(1,:) +1;

%actualSearch = min(size(im2),maxSearchSize*ones(1,ndims(im2)));

for d = 1:dims
    subsCell2{d} = searchBounds_RC(1,d):searchBounds_RC(2,d);
end

refStruct = struct('type','()','subs',{subsCell2});
ncvMatrixROI = subsref( ncvMatrix, refStruct);

%% get the best ncv
[maxNCV,I] = max(ncvMatrixROI(:));
ncvCoords_RC = Utils.IndToCoord(size(ncvMatrixROI),I);

%% return the best coordinate
deltas_RC = ncvCoords_RC - fullOrgin_RC;

if (length(deltas_RC)<3)
    deltas_RC = [deltas_RC,0];
end

if (showDecisionSurf)
    framesDir = 'surfFrames';
    if (saveFrames && ~exist(framesDir,'dir'))
        mkdir(framesDir);
    end
    [X,Y] = meshgrid(1:size(ncvMatrix,2),1:size(ncvMatrix,1));
    X = X - (size(im2,2) + orginCoords_RC(2));
    Y = Y - (size(im2,1) + orginCoords_RC(1));
    [fullmaxNCV,I] = max(ncvMatrix(:));
    fullncvCoords_RC = Utils.IndToCoord(size(ncvMatrix),I);
    fullncvCoords_RC = fullncvCoords_RC - (size(im2) + orginCoords_RC);
    
    figure
    
    subplot(1,2,1);
    if (dims<3)
        surf(X,Y,ncvMatrix,'LineStyle','none');
        nvcZ = 0;
    else
        surf(X,Y,ncvMatrix(:,:,ncvCoords_RC(3)),'LineStyle','none');
        nvcZ = ncvCoords_RC(3) - (size(im2,3) + orginCoords_RC(3));
    end
    hold on
    text(fullncvCoords_RC(2),fullncvCoords_RC(1),fullmaxNCV,...
        sprintf(' %.3f @ \\Delta (%dx,%dy,%dz,%d\\lambda,%.2f\\circ)',fullmaxNCV,fullncvCoords_RC(2),fullncvCoords_RC(1),nvcZ,chan,ang),...
        'Color','r','BackgroundColor','k','VerticalAlignment','bottom');
    xlabel('Delta X')
    ylabel('Delta Y')
    zlabel('Normalized Covariance')
    title('Full Search Space')
    
    if (saveFrames)
        zlim([-.3,1.0]);
    end
    
    subplot(1,2,2);
    [X,Y] = meshgrid(1:size(ncvMatrixROI,2),1:size(ncvMatrixROI,1));
    X = X - fullOrgin_RC(2);
    Y = Y - fullOrgin_RC(1);
    if (dims<3)
        surf(X,Y,ncvMatrixROI,'LineStyle','none');
    else
        surf(X,Y,ncvMatrixROI(:,:,ncvCoords_RC(3)),'LineStyle','none');
    end
    hold on
    text(deltas_RC(2),deltas_RC(1),maxNCV,...
        sprintf(' %.3f @ \\Delta (%dx,%dy,%dz,%d\\lambda,%.2f\\circ)',maxNCV,deltas_RC(2),deltas_RC(1),deltas_RC(3),chan,ang),...
        'Color','r','BackgroundColor','k','VerticalAlignment','bottom');
    xlabel('Delta X')
    ylabel('Delta Y')
    zlabel('Normalized Covariance')
    title('Restricted Search Space (ROI)')
    if (saveFrames)
        zlim([-.3,1.0]);
    end
    
    set(gcf,'units','normalized','OuterPosition',[0,0,1,0.5]);
    drawnow
    if (saveFrames)
        f = getframe(gcf);
        dList = dir(fullfile(framesDir,'*.tif'));
        imwrite(f.cdata,fullfile('surfFrames',sprintf('%04d.tif',length(dList)+1)),'compression','lzw');
        close(gcf);
    end
end
end