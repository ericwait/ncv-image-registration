function tifNamePatternFix(root)
%Lever needs a file pattern of
% DatasetName_c%d_t%03d_z%03d.tif
% all three fields should start at 1 not 0

fileList = dir(fullfile(root,'*.tif*'));

if isempty(fileList)
    return
end

underScores = strfind(fileList(1).name,'_');
zLoc = strfind(fileList(1).name,'_z');
cLoc = strfind(fileList(1).name,'_c');
chString = isletter(fileList(1).name(cLoc+2));
tLoc = strfind(fileList(1).name,'_t');
extLoc = strfind(fileList(1).name,'.tif');
datasetName = fileList(1).name(1:underScores(1)-1);

if ~isempty(zLoc)
    zNumberLoc = underScores(underScores>zLoc);
    if isempty(zNumberLoc)
        zNumberLoc = extLoc;
    end
    zNumberLoc = [zLoc+2 zNumberLoc-1];
    
    if 0==str2double(fileList(1).name(zNumberLoc(1):zNumberLoc(2)))
        zAdd = 1;
    else
        zAdd = 0;
    end
else
    zNumberLoc = [];
end

if ~isempty(tLoc)
    tNumberLoc = underScores(underScores>tLoc);
    if isempty(tNumberLoc)
        tNumberLoc = extLoc;
    end
    tNumberLoc = [tLoc+2 tNumberLoc-1];
    
    if 0==str2double(fileList(1).name(tNumberLoc(1):tNumberLoc(2)))
        tAdd = 1;
    else
        tAdd = 0;
    end
else
    tNumberLoc = [];
end

if ~isempty(cLoc)
    cNumberLoc = underScores(underScores>cLoc);
    if isempty(cNumberLoc)
        cNumberLoc = extLoc;
    end
    
    if chString
        cNumberLoc = [cLoc+3 cNumberLoc-1];
    else
        cNumberLoc = [cLoc+2 cNumberLoc-1];
    end
    
    if 0==str2double(fileList(1).name(cNumberLoc(1):cNumberLoc(2)))
        cAdd = 1;
    else
        cAdd = 0;
    end
else
    cNumberLoc = [];
end

for i=1:length(fileList)
    if ~isempty(strfind(fileList(i).name,'LUT'))
        continue
    end
    
    if isempty(cNumberLoc)
        c = 1;
    else
        c = str2double(fileList(i).name(cNumberLoc(1):cNumberLoc(2))) + cAdd;
    end
    
    if isempty(tNumberLoc)
        t = 1;
    else
        t = str2double(fileList(i).name(tNumberLoc(1):tNumberLoc(2))) + tAdd;
    end
    if isempty(zNumberLoc)
        z = 1;
    else
        z = str2double(fileList(i).name(zNumberLoc(1):zNumberLoc(2))) + zAdd;
    end
    
    newFileName = sprintf('%s\\%s_c%d_t%04d_z%04d.tif',root,datasetName,c,t,z);
    oldFileName = fullfile(root,fileList(i).name);
    
    if strcmpi(newFileName,oldFileName)
        continue
    end
    
    movefile(oldFileName,newFileName);
end
end