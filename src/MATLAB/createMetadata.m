function createMetadata(root,datasetName)
global imageData;

fileList = dir( fullfile([root '\'],'*.xml'));
for i=1:length(fileList)
    parseXML([root '\' fileList(i).name]);
    
    if imageData.NumberOfFrames>0
        fileHandle = fopen([root '\' datasetName '.txt'],'wt');
        fprintf(fileHandle,'DatasetName:%s\n',datasetName);
        fprintf(fileHandle,'NumberOfChannels:%d\n',imageData.NumberOfChannels);
        fprintf(fileHandle,'NumberOfFrames:%d\n',imageData.NumberOfFrames);
        fprintf(fileHandle,'XDimension:%d\n',imageData.XDimension);
        fprintf(fileHandle,'YDimension:%d\n',imageData.YDimension);
        fprintf(fileHandle,'ZDimension:%d\n',imageData.ZDimension);
        fprintf(fileHandle,'XPixelPhysicalSize:%f\n',imageData.XPixelPhysicalSize);
        fprintf(fileHandle,'YPixelPhysicalSize:%f\n',imageData.YPixelPhysicalSize);
        fprintf(fileHandle,'ZPixelPhysicalSize:%f\n',imageData.ZPixelPhysicalSize);
        fprintf(fileHandle,'XPosition:%f\n',imageData.XPosition);
        fprintf(fileHandle,'YPosition:%f\n',imageData.YPosition);
        fclose(fileHandle);
    end
end
end

% 
% cent = [0.017247,0.055831;
% 0.016842,0.055831;
% 0.017361,0.056226;
% 0.016942,0.056226;
% 0.016244,0.056226;
% 0.017449,0.056634;
% 0.017044,0.056634;
% 0.016633,0.056634;
% 0.016230,0.056634;
% 0.017580,0.057026;
% 0.017173,0.057026;
% 0.016793,0.057026;
% 0.016390,0.057026;
% 0.016064,0.057026;
% 0.017574,0.057440;
% 0.017158,0.057440;
% 0.016723,0.057440;
% 0.016341,0.057440;
% 0.017557,0.057875;
% 0.017148,0.057875;
% 0.016714,0.057875;
% 0.016436,0.057875;
% 0.017524,0.058243;
% 0.017110,0.058243;
% 0.016679,0.058243;
% 0.017501,0.058647;
% 0.017079,0.058647;
% 0.016716,0.058647;
% 0.016471,0.058647;
% 0.017353,0.059079;
% 0.016944,0.059079;
% 0.016537,0.059079;
% 0.016143,0.059079;
% 0.017059,0.059497;
% 0.016664,0.059497;
% 0.016256,0.059497;
% 0.016775,0.059886;
% 0.016394,0.059862;
% 0.016643,0.056226;
% 0.016633,0.056634];
% 
% cent = cent*1000000;
% figure
% hold on
% for i=1:length(cent)
%     rectangle('Position',[cent(i,2)-456.3/2,cent(i,1)-456.3/2,456.3,456.3]);
%     text(cent(i,2),cent(i,1),num2str(i));
% end
