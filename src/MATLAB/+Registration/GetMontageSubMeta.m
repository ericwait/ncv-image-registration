function [ imageDatasets, datasetName ] = GetMontageSubMeta( listPath )
%GETMONTAGESUBMETA Takes a path to the list text file that ....
%   Fill this in

imageDatasets = [];
datasetName = '';

if (~exist('listPath','var') || isempty(listPath))
    [fileName,pathName,~] = uigetfile('.txt');
    warning('No data read');
    if fileName==0, return, end
else
    [pathName,fileName,ext] = fileparts(listPath);
    fileName = [fileName ext];
end

fHand = fopen(fullfile(pathName,fileName),'rt');
dirNames = textscan(fHand,'%s','delimiter','\n');
fclose(fHand);

datasetName = [];
str1 = dirNames{1}{1};
for i=length(str1):-1:1
    bmatch = strncmpi(str1(1:i),dirNames{1},i);
    if nnz(bmatch)==length(dirNames{1})
        datasetName = str1(1:i);
        break
    end
end

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
            [imageDatasets,~] = readMetadata(metaFilePath);
        else
            [imageDatasets(end+1),~] = readMetadata(metaFilePath);
        end
    end
end
end

