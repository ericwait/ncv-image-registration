function UpdateXYviewer(im1,im2,normCovar,curDeltaX,curDeltaY,curDeltaZ)
global Fig MaxCovar SubImBest1 SubImBest2
persistent frame

if (isempty(frame))
    frame = 1;
end

if (MaxCovar<normCovar)
    MaxCovar = normCovar;
    imagesc(max(im1,[],3),'Parent',SubImBest1)
    colormap(SubImBest1,'gray')
    axis(SubImBest1,'image')
    imagesc(max(im2,[],3),'Parent',SubImBest2)
    colormap(SubImBest2,'gray')
    axis(SubImBest2,'image')
    titleText = sprintf('Best Deltas (%d,%d,%d):%1.3f',curDeltaX,curDeltaY,curDeltaZ,normCovar);
    title(SubImBest1,titleText);
    title(SubImBest2,titleText);
end

if (~exist('movie','dir'))
    mkdir('movie');
end

%if (mod(curDeltaY,5)==0)
    drawnow
%end

fileName = fullfile('movie',sprintf('ncvIter_%d.tif',frame));
saveas(Fig,fileName);

frame = frame +1;
end
