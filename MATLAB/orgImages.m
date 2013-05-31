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
        fprintf('\nMaking dir %s',[root '\' volumeName ]);
        mkdir(root,volumeName);
    else
        try
            fprintf('.');
            movefile([root '\' volumeName '*'], [root '\' volumeName]);
        catch
            %this seems to error eventhough it worked :-/
%             fprintf(1,'error moving %s\n',[root '\' volumeName '*']);
        end
    end
    createMetadata([root '\' volumeName], volumeName);
    tifNamePatternFix([root '\' volumeName]);
    fileList = dir(fullfile(root,'*.tif'));
end
fprintf('\nDone\n');
end