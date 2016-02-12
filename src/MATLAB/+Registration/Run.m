function Run(listPath,minOverlap,maxSearchSize,unitFactor)
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
if (~exist('unitFactor','var'))
    unitFactor = [];
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
            imageDatasets(i).MinPos = [0,0,0];
            imageDatasets(i).MaxPos = [0,0,0];
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

visualizor = strcmp(refine,'Refine W/ Visualizer');

visualize = questdlg('Would you like to see the NCV results?','Results Visualizer','Yes','No','Visualize Only','No');
if (isempty(visualize)), return, end

visualizor = visualizor || strcmp(visualize,'Yes') || strcmp(visualize,'Visualize Only');

combineHere = questdlg('Would you like to make the final montage on this computer?','Combine Here?','Yes','No','No');

%% Calculate the overlaps if they don't exist of the user wants them recalculated
if (strcmp(refine,'Refine') || strcmp(refine,'Refine W/ Visualizer'))
    prefix = [datasetName '_Montage_wDelta'];
    imageDatasets = Registration.CreateDeltas(imageDatasets,minOverlap,maxSearchSize, unitFactor, visualizor);
    [~,imageDatasets] = Registration.Results.ReadDeltaData(pathName,imageDatasets);
elseif (0==deltasPresent)
    prefix = [datasetName '_Montage'];
end

outPath = MicroscopeData.Helper.CreateUniqueWordedPath(fullfile(pathName,prefix,datasetName));
lastSlash = find(outPath=='\',1,'last');
if (lastSlash==length(outPath))
    lastSlash = find(outPath=='\',2,'last');
    lastSlash = lastSlash(2);
end
outPath = outPath(1:lastSlash);

if (strcmp(combineHere,'Yes'))
    %% Start a log dirctory
    logDir = fullfile(imageDatasets(1).imageDir,'..','_GraphLog');
    if (~exist(logDir,'dir'))
        mkdir(logDir);
    end
    
    %% create a dirctory for the new images
    if ~isdir(outPath)
        mkdir(outPath);
    end
    
    %% Get the final dimensions
    [ imageData, minPos_XY, maxPos_XY ] = Registration.Overlap.GetFinalSize(imageDatasets,datasetName);
    tmpImageData = imageData;
    tmpImageData.DatasetName = strtrim(MicroscopeData.Helper.SanitizeString(tmpImageData.DatasetName));
    datasetName = tmpImageData.DatasetName;
    
    %% Get the image class to make memory for the new image
    imClass = MicroscopeData.GetImageClass(imageDatasets(1));
    
    %% Combine each channel and write it out due to memory constraints
    imMIP = zeros(maxPos_XY(2),maxPos_XY(1),1,imageData.NumberOfChannels,imClass);
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
                % Get the image data
                [nextIm,~] = MicroscopeData.ReaderParZ(imageDatasets(datasetIdx),[],chan,[],[],false,true);
                
                % Calculate where this sub image starts in the full image
                startPos_xy = imageDatasets(datasetIdx).MinPos-minPos_XY+1;
                if (ndims(nextIm)==2)
                    imSize_xy = [Utils.SwapXY_RC(size(nextIm)),1];
                else
                    imSize_xy = Utils.SwapXY_RC(size(nextIm));
                end
                endPos_xy = startPos_xy + min(imageDatasets(datasetIdx).Dimensions,imSize_xy)-1;
                
                outRoi = outImage(startPos_xy(2):endPos_xy(2),startPos_xy(1):endPos_xy(1),startPos_xy(3):endPos_xy(3));
                
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
                outImage(startPos_xy(2):endPos_xy(2),startPos_xy(1):endPos_xy(1),startPos_xy(3):endPos_xy(3)) = nextIm;
                clear nextIm
                
                % Create the colored output image if requested by the user
                if (strcmp(visualize,'No')==0)
                    outImageColor(startPos_xy(2):endPos_xy(2),startPos_xy(1):endPos_xy(1),startPos_xy(3):endPos_xy(3)) =...
                        ones(imageDatasets(datasetIdx).Dimensions(2),imageDatasets(datasetIdx).Dimensions(1),imageDatasets(datasetIdx).Dimensions(3))*datasetIdx;
                end
            end
            
            % Update the user with our progress
            cp.PrintProgress(datasetIdx);
        end
        
        imMIP(:,:,1,chan) = max(outImage,[],3);
        
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
            tmpImageData.Dimensions = Utils.SwapXY_RC(imageData.Dimensions);
        end
        
        % Save out the result
        if (strcmp(visualize,'Visualize Only')==0)
            imwrite(imMIP(:,:,1,chan),fullfile(outPath, ['_' datasetName sprintf('_c%02d_t%04d.tif',chan,1)]),'tif','Compression','lzw');
            MicroscopeData.Writer(outImage,outPath,tmpImageData,[],chan);
        end
        
        % Save a smoothed version
        outImage = Cuda.Mex('ContrastEnhancement',outImage,[75,75,75],[3,3,3]);
        MicroscopeData.Writer(outImage,fullfile(outPath,'Smoothed\'),tmpImageData,[],chan);
        
        % Clean up this channel
        clear outImage;
        if (strcmp(visualize,'No')~=0)
            clear outImageColor;
        end
        tm = toc(chanStart);
        fprintf('done in %s\n',Utils.PrintTime(tm))
    end
    
    tmpImageData.imageDir = outPath;
    if (~isfield(tmpImageData,'ChannelNames')  || isempty(tmpImageData.ChannelNames) ||...
            ~isfield(tmpImageData,'ChannelColors') || isempty(tmpImageData.ChannelColors))
       [ colors, stainNames ] = MicroscopeData.Colors.GetChannelColors(tmpImageData);
       tmpImageData.ChannelNames = stainNames;
       tmpImageData.ChannelColors = colors;
    end
    MicroscopeData.CreateMetadata(tmpImageData.imageDir,tmpImageData,false);
    
    clear outImageColor
    clear difInd
    
    %% Save out overview results
    if (strcmp(visualize,'Visualize Only')==0)
        % Save a colored maximum intensity version
        imageData.imageDir = fullfile(pathName, [prefix, '\']);
        colorMip = ImUtils.ThreeD.ColorMIP(imMIP,MicroscopeData.Colors.GetChannelColors(imageData));
        imwrite(colorMip,fullfile(outPath,sprintf('_%s_RGB.tif',tmpImageData.DatasetName)),'tif','Compression','lzw');
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
        imwrite(frm.cdata,fullfile(outPath,sprintf('_%s_graph.tif',tmpImageData.DatasetName)),'tif','Compression','lzw');
        close(f);
        clear colorMip
    end    
    
end

%% Done!
tm = toc(totalTime);
fprintf('Completed in %s\n',Utils.PrintTime(tm))
end
