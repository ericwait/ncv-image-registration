function makeSpinMovie(rootDir, vidObj, level,vidRes,holdLastFrameFor)
if ~exist('holdLastFrameFor','var') || isempty(holdLastFrameFor)
    holdLastFrameFor = 0;
end

curDir = fullfile(rootDir,sprintf('x%d',level),'ScreenShots');
spinFiles = dir(fullfile(curDir,'*.bmp'));

for i=1:length(spinFiles)
    im = imread(fullfile(curDir,spinFiles(i).name));
    frm = im2frame(im);
    frm = imresize(frm.cdata,vidRes);
    writeVideo(vidObj,frm);
end

for i=1:holdLastFrameFor
    writeVideo(vidObj,frm);
end

end