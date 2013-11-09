vidObj = VideoWriter('22moSVZ.mp4','MPEG-4');
vidObj.FrameRate = 60;
vidObj.Quality = 100;

open(vidObj);

while (true)
    dirPath = uigetdir();
    if (dirPath==0)
        break;
    end
    
    disp(dirPath);
    
    dList = dir(dirPath);
    for i=3:length(dList)
        im = imread(fullfile(dirPath,dList(i).name));
        frm = im2frame(im);
        writeVideo(vidObj,frm);
    end
end

close(vidObj);