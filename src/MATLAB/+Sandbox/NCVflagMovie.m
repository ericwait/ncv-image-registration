global imSz ncv ncvROI originCoords_RC fullOrigin_RC

imSz = size(im2);
imSz = imSz([1,2]);
originCoords_RC = orginCoords_RC([1,2]);
fullOrigin_RC = fullOrgin_RC([1,2]);
ncvMatrix = ncv;
ncvMatrixROI = ncvROI;

mn = min(ncvMatrix(:));
[mx,I] = max(ncvMatrix(:));
mxZ = Utils.IndToCoord(size(ncvMatrix),I);
mxZ = mxZ(3);

vidObj = VideoWriter('iterReg.mp4', 'MPEG-4');
vidObj.Quality = 100;
vidObj.FrameRate = 20;
open(vidObj);

f = figure;
set(f,'unit','pixel','position',[100,100,1920,1080]);
set(f,'Color',[.2 .2 .2]);

ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);

mTextBox = uicontrol('style','text');
set(mTextBox,'unit','normalized','position',[0.47,0.02,0.08,0.05],'BackgroundColor',[.2,.2,.2],'ForegroundColor',[0.7,0.7,0.7],'FontSize',24);

curNcvMatrix = ncvMatrix(:,:,1);
[X,Y] = meshgrid(1:size(curNcvMatrix,2),1:size(curNcvMatrix,1));
Xfull = X - (size(im2,2) + originCoords_RC(2));
Yfull = Y - (size(im2,1) + originCoords_RC(1));

curNcvMatrixROI = ncvMatrixROI(:,:,1);
[X,Y] = meshgrid(1:size(curNcvMatrixROI,2),1:size(curNcvMatrixROI,1));
Xroi = X - fullOrigin_RC(2);
Yroi = Y - fullOrigin_RC(1);

for z=1:size(ncv,3)-1
    surf(Xfull,Yfull,ncvMatrix(:,:,z),'LineStyle','none','Parent',ax1);
    set(ax1,'color',[.3,.3,.3],'XColor',[0.7,0.7,0.7],'YColor',[0.7,0.7,0.7],'ZColor',[0.7,0.7,0.7]);
    zlim(ax1,[mn,mx])    
    xlabel(ax1,'Delta X','Color',[0.7,0.7,0.7])
    ylabel(ax1,'Delta Y','Color',[0.7,0.7,0.7])
    zlabel(ax1,'Normalized Covariance','Color',[0.7,0.7,0.7])
    title(ax1,'Full Search Space','Color',[0.7,0.7,0.7])

    surf(Xroi,Yroi,ncvMatrixROI(:,:,z),'LineStyle','none','parent',ax2);
    set(ax2,'color',[.3,.3,.3],'XColor',[0.7,0.7,0.7],'YColor',[0.7,0.7,0.7],'ZColor',[0.7,0.7,0.7]);
    zlim(ax2,[mn,mx])
    xlabel(ax2,'Delta X','Color',[0.7,0.7,0.7])
    ylabel(ax2,'Delta Y','Color',[0.7,0.7,0.7])
    zlabel(ax2,'Normalized Covariance','Color',[0.7,0.7,0.7])
    title(ax2,'Restricted Search Space (ROI)','Color',[0.7,0.7,0.7])
    
    label = sprintf('Z = %d',z-fullOrgin_RC(3));
    set(mTextBox,'string',label);
    
    drawnow
    
    writeVideo(vidObj,getframe(f));
    
    if (z==mxZ)
        hold(ax1,'on')
        curNcvMatrix = ncvMatrix(:,:,z);
        [fullmaxNCV,I] = max(curNcvMatrix(:));
        fullncvCoords_RC = Utils.IndToCoord(size(curNcvMatrix),I);
        fullncvCoords_RC = fullncvCoords_RC - (imSz + originCoords_RC);
        text(fullncvCoords_RC(2),fullncvCoords_RC(1),fullmaxNCV,...
            sprintf(' \\Delta (%d,%d,%d,%d):%.3f',fullncvCoords_RC(2),fullncvCoords_RC(1),z,chan,fullmaxNCV),...
            'Color','r','BackgroundColor',[.2 .2 .2],'VerticalAlignment','bottom','Parent',ax1);
        hold(ax1,'off')
        
        hold(ax2,'on')
        curNcvMatrixROI = ncvMatrixROI(:,:,z);
        [fullmaxNCV,I] = max(curNcvMatrixROI(:));
        fullncvCoords_RC = Utils.IndToCoord(size(curNcvMatrixROI),I);
        fullncvCoords_RC = fullncvCoords_RC - fullOrigin_RC;
        text(fullncvCoords_RC(2),fullncvCoords_RC(1),fullmaxNCV,...
            sprintf(' \\Delta (%d,%d,%d,%d):%.3f',fullncvCoords_RC(2),fullncvCoords_RC(1),nvcZ,chan,fullmaxNCV),...
            'Color','r','BackgroundColor',[.2 .2 .2],'VerticalAlignment','bottom','Parent',ax2);
        hold(ax2,'off')
        
        fr = getframe(f);
        for i=1:vidObj.FrameRate
            writeVideo(vidObj,fr);
        end
    end
end

close(vidObj);
