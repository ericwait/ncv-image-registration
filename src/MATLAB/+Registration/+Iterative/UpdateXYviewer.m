function UpdateXYviewer(im1,im2,normCovar,curDeltaX,curDeltaY,curDeltaZ)
global Fig MaxCovar SubImBest1 SubImBest2
persistent frame preDeltaX bestY

if (isempty(frame))
    frame = 1;
end

if (isempty(preDeltaX))
    preDeltaX = curDeltaX;
end

if (isempty(bestY))
    bestY = curDeltaY;
end

if (MaxCovar<normCovar)
    g = colormap(SubImBest1,'gray');
    gB = brighten(g,0.3);
    MaxCovar = normCovar;
    imagesc(max(im1,[],3),'Parent',SubImBest1)
    colormap(SubImBest1,gB)
    gB = brighten(g,0.2);
    axis(SubImBest1,'image')
    imagesc(max(im2,[],3),'Parent',SubImBest2)
    colormap(SubImBest2,gB);
    axis(SubImBest2,'image')
    titleText = sprintf('Best Deltas (%d,%d,%d):%1.3f',curDeltaX,curDeltaY,curDeltaZ,normCovar);
    title(SubImBest1,titleText,'Color',[.75,.75,.75],'FontSize',24);
    title(SubImBest2,titleText,'Color',[.75,.75,.75],'FontSize',24);
    set(SubImBest1,'xtick',[])
    set(SubImBest1,'xticklabel',[])
    set(SubImBest1,'ytick',[])
    set(SubImBest1,'yticklabel',[])
    set(SubImBest2,'xtick',[])
    set(SubImBest2,'xticklabel',[])
    set(SubImBest2,'ytick',[])
    set(SubImBest2,'yticklabel',[])
    
    if (~exist('movie','dir'))
        mkdir('movie');
    end
    drawnow
    
    fileName = fullfile('movie',sprintf('ncvIter_%04d.tif',frame));
    X=getframe(Fig);
    i=X.cdata;
    
    i = i(1:end-113,101:end-100,:);
    imwrite(i,fileName);
    
    frame = frame +1;
    bestY = curDeltaY;
elseif (preDeltaX ~= curDeltaX && curDeltaY==bestY)
    if (~exist('movie','dir'))
        mkdir('movie');
    end
    drawnow
    
    fileName = fullfile('movie',sprintf('ncvIter_%04d.tif',frame));
    X=getframe(Fig);
    i=X.cdata;
    
    i = i(1:end-113,101:end-100,:);
    imwrite(i,fileName);
    
    frame = frame +1;
    preDeltaX = curDeltaX;
    
end
end
