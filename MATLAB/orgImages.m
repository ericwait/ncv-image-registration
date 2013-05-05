function orgImages()
root = uigetdir('','Select root dir');
fileList = dir(fullfile(root,'*.tif'));
while ~isempty(fileList)
    tifName = fileList(1).name;
    tokenInd = strfind(tifName,'_');
    if isempty(tokenInd)
        continue
    end
    
    volumeName = tifName(1:tokenInd(1)-1);
    if ~isdir([root '\' volumeName ])
        mkdir(root,volumeName);
    end
    try
        movefile([root '\' volumeName '*'], [root '\' volumeName]);
    catch
        %this seems to error eventhough it worked :-/
    end
    createMetadata([root '\' volumeName], volumeName);
    tifNamePatternFix([root '\' volumeName]);
    fileList = dir(fullfile(root,'*.tif'));
end
end