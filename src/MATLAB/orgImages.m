function orgImages(root)
if (~exist('root','var') || isempty(root))
    root = uigetdir('','Select root dir');
    if (root==0)
        return
    end
end

fileList = dir(root);
bFilesRenamed = 0;
for i=1:length(fileList)
    if (strcmp(fileList(i).name,'.') || strcmp(fileList(i).name,'..'))
        continue
    end
    if (fileList(i).isdir)        
        if (~isempty(strfind(fileList(i).name,'MetaData')))
            fprintf('Parsing XML...');
            readXMLmetadata(fullfile(root,fileList(i).name));
        else
            orgImages(fullfile(root,fileList(i).name));
        end
    else
       if (~isempty(strfind(fileList(i).name,'_')) && bFilesRenamed==0)
           tifNamePatternFix(root);
           bFilesRenamed = 1;
       end 
    end
end
end