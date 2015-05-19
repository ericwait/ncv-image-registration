function updateXYviewer(im1,im2,normCovar,curDeltaX,curDeltaY,curDeltaZ)
global MaxCovar SubImBest1 SubImBest2

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

if (mod(curDeltaY,5)==0)
    drawnow
end
end
