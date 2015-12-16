root = 'B:\Users\Eric\Documents\Programming\RegistrationImages\KD1 Deep\Mosiac';
fileList = dir(fullfile(root,'*.tif'));

maxC = 0;
maxT = 0;
maxZ = 0;

for i=1:size(fileList,1)
    inds = strfind(fileList(i).name,'_');
    if inds(1)==1
        continue;
    end
    
    underScores = strfind(fileList(i).name,'_');
    zLoc = strfind(fileList(i).name,'_z');
    cLoc = strfind(fileList(i).name,'_c');
    chString = fileList(i).name(cLoc+2);
    tLoc = strfind(fileList(i).name,'_t');
    extLoc = strfind(fileList(i).name,'.tif');
    datasetName = fileList(i).name(1:underScores(1)-1);
    tString = fileList(i).name(tLoc+2:tLoc+5);
    zString = fileList(i).name(zLoc+2:zLoc+5);
    c = str2double(chString);
    t = str2double(tString);
    z = str2double(zString);
    
    maxC = max(maxC,c);
    maxT = max(maxT,t);
    maxZ = max(maxZ,z);
end

image = [];

for i=1:size(fileList,1)
    inds = strfind(fileList(i).name,'_');
    if inds(1)==1
        continue;
    end
    
    underScores = strfind(fileList(i).name,'_');
    zLoc = strfind(fileList(i).name,'_z');
    cLoc = strfind(fileList(i).name,'_c');
    chString = fileList(i).name(cLoc+2);
    tLoc = strfind(fileList(i).name,'_t');
    extLoc = strfind(fileList(i).name,'.tif');
    datasetName = fileList(i).name(1:underScores(1)-1);
    tString = fileList(i).name(tLoc+2:tLoc+5);
    zString = fileList(i).name(zLoc+2:zLoc+5);
    c = str2double(chString);
    t = str2double(tString);
    z = str2double(zString);
    
    im = imread(fullfile(root,fileList(i).name));
    
    if isempty(image)
        image = zeros(size(im,1),size(im,2),maxZ,maxT,maxC,'uint8');
    end
    
    image(:,:,z,t,c) = im;
end

for c=1:maxC
    figure
    imagesc(max(image(:,:,:,1,c),[],3));
    colormap gray
end