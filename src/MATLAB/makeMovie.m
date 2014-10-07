function makeMovie()
%% SET THESE!!!!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
levels = [5,3,2,1,0];
center = [7953 1992];
vidRes = [1080 1920];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% go
[fileName, rootDir] = uigetfile('*.txt');
imageData = readMetaData(fullfile(rootDir,fileName));

%vidObj = VideoWriter(fullfile(rootDir,sprintf('%s.mp4',imageData.DatasetName)),'MPEG-4');
vidObj = VideoWriter(fullfile(rootDir,sprintf('%s.avi',imageData.DatasetName)),'Uncompressed AVI');
vidObj.FrameRate = 60;
%vidObj.Quality = 100;

open(vidObj);

for levelIdx=1:length(levels)-1
    makeSpinMovie(rootDir,vidObj,levels(levelIdx),vidRes);
    makeSmoothZoom(rootDir,vidObj,levels(levelIdx),levels(levelIdx+1),imageData,center,vidRes);
end

makeSpinMovie(rootDir,vidObj,levels(end),vidRes);
makeSpinMovie(rootDir,vidObj,levels(end),vidRes,30);

close(vidObj);

end