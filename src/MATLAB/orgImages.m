function orgImages(root)
if (~exist('root','var') || isempty(root))
    root = uigetdir('','Select root dir');
    if (root==0)
        return
    end
end

fileList = dir(root);
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
    end
end

tifNamePatternFix(root);
end