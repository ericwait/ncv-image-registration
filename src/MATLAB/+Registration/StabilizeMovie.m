function [imOut, imageDataOut,cumulativeDeltas_rc] = StabilizeMovie(imIn,imageDataIn,cumulativeDeltas_rc,unitFactor,minOverlap,maxSearchSize,logFile,showDecisionSurf,visualize, imMask1, imMask2)
%RUN Takes a movie and attempts to stabilize the objects between frames by
% translation of sebsequent frames
%[imOut, imageDataOut,cumulativeDeltas_rc] = StabilizeMovie(imIn,imageDataIn,cumulativeDeltas_rc,unitFactor,minOverlap,maxSearchSize,logFile,showDecisionSurf,visualize, imMask1, imMask2)
%imIn - This should be a 5D image as (x,y,z,chan,time);
%imageDataIn - This should be a well formed metadata structure. See MicroscopeData.GetEmptyMetadata
%The following are optional paramaters:
%       cumulativeDeltas_rc - these are the deltas that will be applied to
%          the sequence without running interframe registration
%       unitFactor - this is a multiplyer to get the position data into the
%           same units as the PixelPhysicalSize
%       minOverlap - this is the 
% default 25
%       movementExtent
% default 100
%       logFile
% default stdOut
%       visualize
% default false
%       imMask1
% default empty
%       imMask2
% default empty


%% check inputs
    if (~exist('cumulativeDeltas_rc','var'))
        cumulativeDeltas_rc = [];
    end
    if (~exist('unitFactor','var'))
        unitFactor = [];
    end
    if (~exist('minOverlap','var'))
        minOverlap = [];
    end
    if (~exist('maxSearchSize','var'))
        maxSearchSize = [];
    end
    if (~exist('logFile','var'))
        logFile = [];
    end
    if (~exist('showDecisionSurf','var'))
        showDecisionSurf = [];
    end
    if (~exist('visualize','var'))
        visualize = [];
    end
    if (~exist('imMask1','var'))
        imMask1 = [];
    end
    if (~exist('imMask2','var'))
        imMask2 = [];
    end
    
    is = tic;
    
%% run the reg
    if (isempty(cumulativeDeltas_rc))
    % calculate the size of the parallel pool needed
    
        numVoxels = prod(imageDataIn.Dimensions);
        w = whos('imIn');
        memNeededBytes = numVoxels*8*8 + 2*w.bytes;
        m = memory;
        pc = parcluster('local');

        numWorkers = floor((m.MemAvailableAllArrays*0.80)/memNeededBytes);
        numWorkers = min(pc.NumWorkers,numWorkers);

        p = gcp('nocreate');
        oldWorkers = 0;
        if (isvalid(p))
            oldWorkers = p.NumWorkers;
        end
        if (oldWorkers==0)
            parpool(numWorkers);
        elseif (oldWorkers>numWorkers)
            delete(p);
            parpool(numWorkers);
        end    
    
        imDt = imageDataIn;
        imDt.NumberOfFrames = 1;

        frameDeltas_xyz = zeros(imageDataIn.NumberOfFrames,5);
        parfor t=1:imageDataIn.NumberOfFrames-1
            curFrame = imIn(:,:,:,:,t);
            curD = imDt;
            curD.Position = [0,0,0];
            %curD.Position = squeeze(posT(1,1,t,:))';
            nextFrame = imIn(:,:,:,:,t+1);
            nextD = imDt;
            nextD.Position = [0,0,0];
            %nextD.Position = squeeze(posT(1,1,t+1,:))';

            [deltaX,deltaY,deltaZ,maxNCV,overlapSize] = Registration.FFT.RegisterTwoImages(curFrame,curD,nextFrame,nextD,unitFactor,minOverlap,maxSearchSize,logFile,false, imMask1, imMask2);

            if (isinf(maxNCV))
                [deltaX,deltaY,deltaZ,maxNCV,overlapSize] = Registration.FFT.RegisterTwoImages(curFrame,curD,nextFrame,nextD,unitFactor,minOverlap,maxSearchSize,logFile,true, imMask1, imMask2);
                title(num2str(t));
            end

            frameDeltas_xyz(t+1,:) = [deltaX,deltaY,deltaZ,maxNCV,overlapSize];
        end

        frameDeltas_rc = Utils.SwapXY_RC(frameDeltas_xyz);
        cumulativeDeltas_rc = cumsum(frameDeltas_rc(:,1:3),1);
    end

%% apply deltas
    maxDelta_rc = max(cumulativeDeltas_rc,[],1);
    minDelta_rc = min(cumulativeDeltas_rc,[],1);
    
    movementExtent_rc = maxDelta_rc + abs(minDelta_rc);
    
    newSize = Utils.SwapXY_RC(imageDataIn.Dimensions) + movementExtent_rc;
    imOut = zeros([newSize,imageDataIn.NumberOfChannels,imageDataIn.NumberOfFrames],'like',imIn);
    
    for t=1:imageDataIn.NumberOfFrames
        newPosStart = cumulativeDeltas_rc(t,:) - minDelta_rc +1;
        newPosEnd = newPosStart + imageDataIn.Dimensions([2,1,3]) -1;
        imOut(newPosStart(1):newPosEnd(1),newPosStart(2):newPosEnd(2),newPosStart(3):newPosEnd(3),:,t) = imIn(:,:,:,:,t);
    end
    
    if (exist('oldWorkers','var') && oldWorkers~=0 && numWorkers~=oldWorkers)
        p = gcp('nocreate');
        delete(p);
        parpool(oldWorkers);
    end
    
    imageDataOut = imageDataIn;
    imageDataOut.DatasetName = [imageDataOut.DatasetName,'_registered'];
    sz = size(imOut);
    imageDataOut.Dimensions = sz([2,1,3]);
    sec = toc(is);
    fprintf('Image Stabilization Took: %s, avg: %s\n',Utils.PrintTime(sec),Utils.PrintTime(sec/size(imIn,5)));
end

