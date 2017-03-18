function [ imageDatasets, datasetName ] = GetMontageSubMeta( listPath )
%GETMONTAGESUBMETA Takes a path to the list text file that ....
%   Fill this in

imageDatasets = [];
datasetName = '';

if (~exist('listPath','var') || isempty(listPath))
    [fileName,pathName,~] = uigetfile({'*.txt';'*.*'});
    if fileName==0
        warning('No data read');
        return
    end
else
    [pathName,fileName,ext] = fileparts(listPath);
    fileName = [fileName ext];
end

[~,~,ext] = fileparts(fileName);

if (strcmpi(ext,'.txt'))
    fHand = fopen(fullfile(pathName,fileName),'rt');
    dirNames = textscan(fHand,'%s','delimiter','\n');
    fclose(fHand);
        
    for i=1:length(dirNames{1})
        [~,~,ext] = fileparts(dirNames{1}{i});
        if (strcmpi(ext,'.txt'))
            metaFilePath = fullfile(pathName,dirNames{1}{i});
        elseif (strcmpi(ext,'.json'))
            metaFilePath = fullfile(pathName,dirNames{1}{i},[dirNames{1}{i},'.json']);
        else
            metaFilePath = fullfile(pathName,dirNames{1}{i},[dirNames{1}{i},'.txt']);
            if (~exist(metaFilePath,'file'))
                metaFilePath = fullfile(pathName,dirNames{1}{i},[dirNames{1}{i},'.json']);
                if (~exist(metaFilePath,'file'))
                    metaFilePath = [];
                end
            end
        end
        
        if (~isempty(metaFilePath))
            if isempty(imageDatasets)
                [imageDatasets,~] = MicroscopeData.ReadMetadata(metaFilePath);
            else
                [imageDatasets(end+1),~] = MicroscopeData.ReadMetadata(metaFilePath);
            end
        end
    end
else
    [~,imageDatasets] = MicroscopeData.Original.ConvertData(pathName,fileName,[],true,true,false,true);
    imageDatasets = cellfun(@(x)(x),imageDatasets);
    for i=1:length(imageDatasets)
        imageDatasets(i).imageDir = fullfile(pathName,imageDatasets(i).DatasetName);
    end
end

datasetName = '';
dNames = arrayfun(@(x)(x.DatasetName),imageDatasets,'UniformOutput',false);
lengths = cellfun(@(x)(length(x)),dNames);
for i=1:min(lengths)
    letter = dNames{1}(i);
    match = cellfun(@(x)(strcmp(x(i),letter)),dNames);
    if (all(match))
        datasetName = [datasetName,letter];
    else
        break
    end
end
datasetName = MicroscopeData.Helper.SanitizeString(datasetName);
end

