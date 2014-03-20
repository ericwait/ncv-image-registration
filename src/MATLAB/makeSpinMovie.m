function makeSpinMovie(rootDir, vidObj, level)

curDir = fullfile(rootDir,'movie',sprintf('x%d',level),'spin');
spinFiles = dir(fullfile(curDir,'closer*.bmp'));

for i=1:length(spinFiles)
    im = imread(fullfile(curDir,spinFiles(i).name));
    frm = im2frame(im);
    writeVideo(vidObj,frm);
end

end