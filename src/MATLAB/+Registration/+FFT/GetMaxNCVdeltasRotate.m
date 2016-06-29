function [deltasCenter_RC,maxNCV] = GetMaxNCVdeltasRotate(im1,im1BW,im2,im2BW,maxAngle,incAngle,minOverlapVolume,maxSearchSize,orginCoords_RC,showDecisionSurf,chan,saveFrames)
if (isempty(im1BW))
    im1BW = true(size(im1));
end
if (isempty(im2BW))
    im2BW = true(size(im2));
end
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
if (~exist('incAngle','var') || isempty(incAngle))
    incAngle = 1;
end
if (~exist('saveFrames','var'))
    saveFrames = [];
end

deltasCenter_RC = convertToCenter(size(im1),size(im2),[0,0,0,0]);
maxNCV = -inf;

for ang=-maxAngle:incAngle:maxAngle
    im2a = imrotate(im2,ang,'bicubic');
    im2BWa = imrotate(im2BW,ang,'bicubic')>0;    
    [curDeltas_RC,curMaxNCV] = Registration.FFT.GetMaxNCVdeltas(im1,im2a,minOverlapVolume,maxSearchSize,orginCoords_RC,showDecisionSurf,chan,ang,im1BW,im2BWa,saveFrames);
    if (maxNCV < curMaxNCV)
        maxNCV = curMaxNCV;
        curDeltasCntr = convertToCenter(size(im1),size(im2a),curDeltas_RC);
        deltasCenter_RC = [curDeltasCntr,ang];
    end
end
end

function deltasOut_rc = convertToCenter(imSize1, imSize2, deltas_rc)
deltasOut_rc = zeros(size(deltas_rc));

if (numel(imSize1)<3)
    imSize1 = [imSize1,ones(1,3-numel(imSize1))];
end
if (numel(imSize2)<3)
    imSize2 = [imSize2,ones(1,3-numel(imSize2))];
end

for d=1:3
    im1center = imSize1(d)/2;
    im2center = imSize2(d)/2;
    
    if (deltas_rc(d)<0)
        % The origin of two is before one
        im1center = im1center - deltas_rc(d);
    elseif (deltas_rc(d)>0)
        % The origin of one is before two
        im2center = im2center + deltas_rc(d);
    else
        % The origin of one and two are aligned
    end
    
    deltasOut_rc(d) = im1center - im2center;
end

end