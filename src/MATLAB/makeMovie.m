function makeMovie()
%% SET THESE!!!!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
levels = [7,4,2,1,0];
center = [7953 1992];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% go
[fileName, rootDir] = uigetfile('*.txt');
imageData = readMetaData(fullfile(rootDir,fileName));

vidObj = VideoWriter(fullfile(rootDir,sprintf('%s.mp4',imageData.DatasetName)),'MPEG-4');
vidObj.FrameRate = 60;
vidObj.Quality = 100;

open(vidObj);

for levelIdx=1:length(levels)-1
    makeSpinMovie(rootDir,vidObj,levels(levelIdx));
    makeSmoothZoom(rootDir,vidObj,levels(levelIdx),levels(levelIdx+1),imageData,center);
end

makeSpinMovie(rootDir,vidObj,levels(end));

close(vidObj);

end