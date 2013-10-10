function [xOffset yOffset maxCorrLU] = registerTwoImages(staticImage, otherImage)
global MARGIN

% staticMu = mean(staticImage(:));
% otherMu = mean(otherImage(:));
% staticSig = std(staticImage(:));
% otherSig = std(otherImage(:));

corrMapRD = zeros(MARGIN,MARGIN);
corrMapLU = zeros(MARGIN,MARGIN);
mr = MARGIN;
for deltaX=1:mr
    if 0==mod(deltaX,10)
        xTicId = tic;
    end
    parfor deltaY=1:mr
%         yTicId = tic;
        %move the otherImage right and down this much
        A = staticImage(deltaY:end,deltaX:end,:);
        B = otherImage(deltaY:end,deltaX:end,:);
%         c = max(A,[],3);figure(1), imagesc(c),axis equal,colormap gray
%         d = max(B,[],3);figure(2), imagesc(d),axis equal,colormap gray
        corrMapRD(deltaY,deltaX) = calcCorrelation(A,B);
%         corrMapRD(deltaY,deltaX) = calcCorrelation(A,B,staticMu,otherMu,staticSig,otherSig);
        %move the otherImage left and up this much
        A = staticImage(1:end-deltaY+1,1:end-deltaX+1,:);
        B = otherImage(1:end-deltaY+1,1:end-deltaX+1,:);
%         c = max(A,[],3);figure(1), imagesc(c),axis equal,colormap gray
%         d = max(B,[],3);figure(2), imagesc(d),axis equal,colormap gray
        corrMapLU(deltaY,deltaX) = calcCorrelation(A,B);
%         corrMapLU(deltaY,deltaX) = calcCorrelation(A,B,staticMu,otherMu,staticSig,otherSig);
%         tY = toc(yTicId);
%         if(mod(deltaY,10)==0)
%             fprintf(1,'(%d,%f) ',deltaY,tY);
%         end
    end
    if 0==mod(deltaX,10)
        tX = toc(xTicId);
        fprintf(1,'%d:%f,',deltaX,tX);
    end
end

maxCorrRD = max(corrMapRD(:));
maxCorrLU = max(corrMapLU(:));
globalMax = max(maxCorrRD,maxCorrLU);

if globalMax==0
    xOffset = MARGIN;
    yOffset = MARGIN;
else
    if globalMax==maxCorrRD
        %we have an offset going right and down
        [yOffset xOffset] = ind2sub([MARGIN MARGIN],find(corrMapRD==maxCorrRD));
        yOffset = yOffset-1;
        xOffset = xOffset-1;%because of 1 index means no change
    else
        %we have an offset going left and up
        [yOffset xOffset] = ind2sub([MARGIN MARGIN],find(corrMapLU==maxCorrLU));
        xOffset = 1-xOffset;
        yOffset = 1-yOffset;%because of 1 index means no change
    end
end
% fprintf(1,'\n(%f,%f)\n',xOffset,yOffset);
end

function correlation = calcCorrelation(A,B,aMu,bMu,aSig,bSig)
if ~exist('aMu','var')
    aMu = mean(A(:));
end
if ~exist('bMu','var')
    bMu = mean(B(:));
end
if ~exist('aSig','var')
    aSig = std(A(:));
end
if ~exist('bSig','var')
    bSig = std(B(:));
end

mult = (A-aMu).*(B-bMu);
correlation = (sum(mult(:)))/(aSig*bSig);
end