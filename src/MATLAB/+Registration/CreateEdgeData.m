function CreateEdgeData( dirs, names, logFile, minOverlap,maxSearchSize,unitFactor,visualize)
%CREATEEDGEDATA Summary of this function goes here
%   Detailed explanation goes here

%% Create mutexes and local data
[cleanupObj,fileMap] = Threading.InitCleanupData();

ed = struct(...
    'i',0,...
    'j',0,...
    'normCovar',-inf,...
    'deltaX',0,...
    'deltaY',0,...
    'deltaZ',0,...
    'overlapSize',0);
            
%% Iterate over all of the images
for i=labindex:numlabs:length(names)
    static = tic;
    im1 = [];
    imageDataset1 = [];
    
    %% Iterate over all of the other edges not expolored
    for j=i+1:length(names)
        % check if the mutex is available
        checkPointPath = fullfile(dirs{i},sprintf('%d_%d.mat',i,j));
        [bExists,~] = Threading.ClaimDataFile(fileMap, checkPointPath);
        
        if (~bExists)
            % Read in the first image only once
            if (isempty(im1))
                [im1,imageDataset1] = MicroscopeData.Reader(fullfile(dirs{i},names{i}),[],[],[],[],false,true);
            end
            
            % Check if the next image overlaps or not
            imageDataset2 = MicroscopeData.ReadMetadata(fullfile(dirs{j},names{j}));
            [roi1,~,minXdist,minYdist] = Registration.Overlap.CalculateOverlapXY(imageDataset1,imageDataset2,unitFactor);
            
            ed.i = i;
            ed.j = j;
            roiVol = max(roi1(4)-roi1(1),1) * max(roi1(5)-roi1(2),1) * max(roi1(6)-roi1(3),1);
                
            % The images do not overlap set the edge to the default value
            if (minXdist>maxSearchSize-minOverlap || minYdist>maxSearchSize-minOverlap || roiVol<maxSearchSize^2)
                ed.normCovar = -inf;
                ed.deltaX = 0;
                ed.deltaY = 0;
                ed.deltaZ = 0;
                ed.overlapSize = 0;
            else
                % These images overlap enough, log that we are going to
                % calculate it
                t = datetime('now');
                fprintf(1,'\t%s-->%s @ %02d:%02d:%02d.%0.0f\n',...
                    imageDataset1.DatasetName,imageDataset2.DatasetName,t.Hour,t.Minute,floor(t.Second),(t.Second-floor(t.Second))*100);
                
                % Read in the second image
                [im2,~] = MicroscopeData.Reader(fullfile(dirs{j},names{j}),[],[],[],[],false,true);
                
                % Run the registration
                if (visualize)
                    [deltaX,deltaY,deltaZ,normCovar,overlapSize] = Registration.Iterative.RegisterTwoImages(im1,imageDataset1,im2,imageDataset2,unitFactor,...
                        minOverlap,maxSearchSize,logFile,visualize,visualize);
                else
                    [deltaX,deltaY,deltaZ,normCovar,overlapSize] = Registration.FFT.RegisterTwoImages(im1,imageDataset1,im2,imageDataset2,unitFactor,...
                        minOverlap,maxSearchSize,logFile,visualize,visualize);
                end
                
                % Collect the results
                ed.normCovar = normCovar;
                ed.deltaX = deltaX;
                ed.deltaY = deltaY;
                ed.deltaZ = deltaZ;
                ed.overlapSize = overlapSize;
            end
            
            % Release the mutex
            Threading.FinalizeDataFile(fileMap, checkPointPath, ed);
        end
    end
    
    %% Finished this image to all adjecent 
    tm = toc(static);
    fHand = fopen(logFile,'at');
    fprintf(fHand,'%s took %s\n',names{i},Utils.PrintTime(tm));
    fprintf('%s took %s\n\n',names{i},Utils.PrintTime(tm))
    fclose(fHand);
end
end
