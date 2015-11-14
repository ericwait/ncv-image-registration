function Run(listPath,minOverlap,maxSearchSize)
%% Check inputs
if (~exist('listPath','var'))
    listPath = [];
end

if (~exist('minOverlap','var') || isempty(minOverlap))
    minOverlap = 25;
end

if (~exist('maxSearchSize','var') || isempty(maxSearchSize))
    maxSearchSize = 100;
end

totalTime = tic;

%% Get all the metadata
[ imageDatasets, datasetName ] = Registration.GetMontageSubMeta(listPath);

if (isempty(imageDatasets))
    warning('No images found!');
    return
end

pathName = fullfile(imageDatasets(1).imageDir,'..');
[deltasPresent,imageDatasets] = Registration.Results.ReadDeltaData(pathName,imageDatasets);
if (deltasPresent==true)
    refine = questdlg('Would you like to use the old registration delta?','Refine Deltas?','Old','Redo','Old');
    if (refine==0)
        return
    elseif (strcmp(refine,'Redo'))
        deltasPresent = 0;
        
        for i=1:length(imageDatasets)
            imageDatasets(i).ParentDelta = 0;
            imageDatasets(i).Children = [];
            imageDatasets(i).xMinPos = 0;
            imageDatasets(i).yMinPos = 0;
            imageDatasets(i).zMinPos = 0;
            imageDatasets(i).xMaxPos = 0;
            imageDatasets(i).yMaxPos = 0;
            imageDatasets(i).zMaxPos = 0;
        end
    end
end

%% Ask the user how they would like to procced now that we know what inforamtion we have

if (deltasPresent==false)
    refine = questdlg('Would you like to refine registration or use microscope data?','Refine Deltas?','Refine','Microscope','Refine W/ Visualizer','Microscope');
    if (refine==0), return, end
else
    prefix = [datasetName '_Montage_wDelta'];
    refine = '';
end

visualize = questdlg('Would you like to see the results?','Results Visualizer','Yes','No','Visualize Only','No');
if (isempty(visualize)), return, end

%% Calculate the overlaps if they don't exist of the user wants them recalculated
if (strcmp(refine,'Refine') || strcmp(refine,'Refine W/ Visualizer'))
    prefix = [datasetName '_Montage_wDelta'];
    imageDatasets = Registration.CreateDeltas(imageDatasets,minOverlap,maxSearchSize,strcmp(refine,'Refine W/ Visualizer'));
    [~,imageDatasets] = Registration.Results.ReadDeltaData(pathName,imageDatasets);
elseif (0==deltasPresent)
    prefix = [datasetName '_Montage'];
end

%% Start a log dirctory
logDir = fullfile(imageDatasets(1).imageDir,'..','_GraphLog');
if (~exist(logDir,'dir'))
    mkdir(logDir);
end

%% create a dirctory for the new images
if ~isdir(fullfile(pathName,prefix))
    mkdir(pathName,prefix);
end

%% Get the final dimensions
[ imageData, minPos_XY, maxPos_XY ] = Registration.Overlap.GetFinalSize(imageDatasets,datasetName);
tmpImageData = imageData;

%% Get the image class to make memory for the new image
imClass = MicroscopeData.GetImageClass(imageDatasets(1));

%% Combine each channel and write it out due to memory constraints
for chan=1:imageData.NumberOfChannels
    chanStart = tic;
    
    % prealocate memory
    outImage = zeros(maxPos_XY(2),maxPos_XY(1),maxPos_XY(3),imClass);
    if (strcmp(visualize,'No')==false)
        outImageColor = zeros(maxPos_XY(2),maxPos_XY(1),maxPos_XY(3),imClass);
    end
    
    % Give the user an indication of were the process is
    fprintf('Chan:%d...',chan);
    cp = Utils.CmdlnProgress(length(imageDatasets),true);
    
    % Place each subimage into the final channel image
    for datasetIdx=1:length(imageDatasets)
        
        % Ensure that this subimage has the current channel
        if (imageDatasets(datasetIdx).NumberOfChannels>=chan)
            
            % Calculate where this sub image starts in the full image
            startXind = imageDatasets(datasetIdx).xMinPos-minPos_XY(1)+1;
            startYind = imageDatasets(datasetIdx).yMinPos-minPos_XY(2)+1;
            startZind = imageDatasets(datasetIdx).zMinPos-minPos_XY(3)+1;
            
            % Get the image data
            [nextIm,~] = MicroscopeData.Reader(imageDatasets(datasetIdx),[],chan,[],[],false,true);
            
            % Get the range of indices that this sub image exist in the
            % large image
            roi_RC = floor([startYind,startXind,startZind,...
                startYind+min(imageDatasets(datasetIdx).YDimension,size(nextIm,1))-1,...
                startXind+min(imageDatasets(datasetIdx).XDimension,size(nextIm,2))-1,...
                startZind+min(imageDatasets(datasetIdx).ZDimension,size(nextIm,3))-1]);
            
            outRoi = outImage(roi_RC(1):roi_RC(4),roi_RC(2):roi_RC(5),roi_RC(3):roi_RC(6));
            
            % Find out which image should exist in the overlap
            % TODO this could be blended better
            difInd = outRoi>0;
            nextSum = sum(sum(sum(nextIm(difInd))));
            outSum = sum(sum(sum(outRoi(difInd))));
            if outSum>nextSum
                nextIm(difInd) = outRoi(difInd);
            end
            clear outRoi
            
            % Set the output image with the sub image
            outImage(roi_RC(1):roi_RC(4),roi_RC(2):roi_RC(5),roi_RC(3):roi_RC(6)) = nextIm;
            clear nextIm
            
            % Create the colored output image if requested by the user
            if (strcmp(visualize,'No')==0)
                outImageColor(startYind:startYind+imageDatasets(datasetIdx).YDimension-1,...
                    startXind:startXind+imageDatasets(datasetIdx).XDimension-1,...
                    startZind:startZind+imageDatasets(datasetIdx).ZDimension-1) = ones(imageDatasets(datasetIdx).YDimension,...
                    imageDatasets(datasetIdx).XDimension,imageDatasets(datasetIdx).ZDimension)*datasetIdx;
            end
        end
        
        % Update the user with our progress
        cp.PrintProgress(datasetIdx);
    end
    
    % Cleanup the progress indicator
    cp.ClearProgress();
    
    % Show the user the output if requested
    if (strcmp(visualize,'Yes') || strcmp(visualize,'Visualize Only'))
        figure,imagesc(max(outImage,[],3)),colormap gray, axis image
        title(sprintf('Cannel:%d',chan),'Interpreter','none','Color','w');
        Registration.Results.TestingDeltas(outImage, outImageColor,imageDatasets,chan,prefix);
    end
    
    % Write out the resuts to a file
    if (~strcmp(visualize,'Visualize Only'))
        [fig,ax] = Registration.Results.TestingDeltas(outImage,[],imageDatasets,chan,prefix);
        set(fig,'Units','normalized','Position',[0 0 1 1]);
        if (size(outImage,1)>size(outImage,2))
            camroll(ax,-90);
        end
        frm = getframe(ax);
        imwrite(frm.cdata,fullfile(logDir,sprintf('%s_c%02d_minSpanTree.tif',datasetName,chan)),'tif','Compression','lzw');
        close(fig);
    end
    
    % Make sure the image is horizontal
    % Rotate it if it is not and there is enough memory
    w = whos('outImage');
    userview = memory;
    if (size(outImage,1)>size(outImage,2) && userview.MemAvailableAllArrays>w.bytes)
        outImage = permute(outImage(end:-1:1,:,:),[2,1,3]);
        tmpImageData.XDimension = imageData.YDimension;
        tmpImageData.YDimension = imageData.XDimension;
    end
    
    % Save out the result
    if (strcmp(visualize,'Visualize Only')==0)
        imwrite(max(outImage,[],3),fullfile(pathName, prefix, ['_' datasetName sprintf('_c%02d_t%04d.tif',chan,1)]),'tif','Compression','lzw');
        MicroscopeData.Writer(outImage,fullfile(pathName, [prefix, '\']),tmpImageData,[],chan);
    end
    
    % Clean up this channel
    clear outImage;
    if (strcmp(visualize,'No')~=0)
        clear outImageColor;
    end
    tm = toc(chanStart);
    fprintf('done in %s\n',Utils.PrintTime(tm))
end

tmpImageData.imageDir = fullfile(pathName, [prefix, '\']);

%% Save out overview results
if (strcmp(visualize,'Visualize Only')==0)
    % Save a colored maximum intensity version
    imageData.imageDir = fullfile(pathName, [prefix, '\']);
    colorMip = ImUtils.ThreeD.ColorMIP(MicroscopeData.Reader(tmpImageData),MicroscopeData.GetChannelColors(imageData));
    imwrite(colorMip,fullfile(pathName,prefix,sprintf('_%s_RGB.tif',tmpImageData.DatasetName)),'tif','Compression','lzw');
    f = figure;
    imagesc(colorMip);%,'Parent',ax);
    ax = get(f,'CurrentAxes');
%     if (tmpImageData.XDimension~=imageData.XDimension)
%         camroll(ax,90);
%     end
    Registration.Results.DrawBoxesLines(f,ax,imageDatasets,0,tmpImageData.DatasetName);
%     if (size(colorMip,1)>size(colorMip,2))
%         camroll(ax,-90);
%     end
    frm = getframe(ax);
    imwrite(frm.cdata,fullfile(pathName,prefix,sprintf('_%s_graph.tif',tmpImageData.DatasetName)),'tif','Compression','lzw');
    close(f);
end

%% Done!
tm = toc(totalTime);
fprintf('Completed in %s\n',Utils.PrintTime(tm))
end