function createEdgeData( dirs, names, logFile, minOverlap,maxSearchSize,visualize)
%CREATEEDGEDATA Summary of this function goes here
%   Detailed explanation goes here

[cleanupObj,fileMap] = Threading.initCleanupData();

ed = struct(...
    'i',0,...
    'j',0,...
    'normCovar',-inf,...
    'deltaX',0,...
    'deltaY',0,...
    'deltaZ',0,...
    'overlapSize',0);
            
for i=labindex:numlabs:length(names)
    static = tic;
    im1 = [];
    imageDataset1 = [];
    
    for j=i+1:length(names)
        checkPointPath = fullfile(fullfile(dirs{i},sprintf('%d_%d.mat',i,j)));
        [bExists,~] = Threading.claimDataFile(fileMap, checkPointPath);
        
        if (~bExists)
            if (isempty(im1))
                [im1,imageDataset1] = tiffReader(fullfile(dirs{i},[names{i},'.txt']),[],[],[],[],[],true);
            end
            
            imageDataset2 = readMetadata(fullfile(dirs{j},[names{j},'.txt']));
            [~,~,minXdist,minYdist] = calculateOverlap(imageDataset1,imageDataset2);
            
            ed.i = i;
            ed.j = j;
                
            if (minXdist>maxSearchSize-minOverlap || minYdist>maxSearchSize-minOverlap)
                ed.normCovar = -inf;
                ed.deltaX = 0;
                ed.deltaY = 0;
                ed.deltaZ = 0;
                ed.overlapSize = 0;
            else
                t = datetime('now');
                fprintf(1,'\t%s-->%s @ %02d:%02d:%02d.%0.0f\n',...
                    imageDataset1.DatasetName,imageDataset2.DatasetName,t.Hour,t.Minute,floor(t.Second),(t.Second-floor(t.Second))*100);
                
                [im2,~] = tiffReader(fullfile(dirs{j},[names{j},'.txt']),[],[],[],[],[],true);
                
                [deltaX,deltaY,deltaZ,normCovar,overlapSize] = registerTwoImagesFTT(im1,imageDataset1,im2,imageDataset2,...
                    minOverlap,maxSearchSize,logFile);
                
                ed.normCovar = normCovar;
                ed.deltaX = deltaX;
                ed.deltaY = deltaY;
                ed.deltaZ = deltaZ;
                ed.overlapSize = overlapSize;
            end
            
            Threading.finalizeDataFile(fileMap, checkPointPath, ed);
        end
    end
    
    tm = toc(static);
    fHand = fopen(logFile,'at');
    fprintf(fHand,'%s took %s\n',names{i},printTime(tm));
    fprintf(1,'%s took %s\n\n',names{i},printTime(tm));
    fclose(fHand);
end

