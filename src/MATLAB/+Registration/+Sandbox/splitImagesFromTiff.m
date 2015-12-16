root = 'D:\Users\Eric\Documents\Programming\Images\22mo2 wmSVZ Unmixed';

fileHandle = fopen(fullfile(root, 'list.txt'));

datasets = {};
i = 1;
ln = fgetl(fileHandle);
while (ischar(ln))
    datasets{i} = ln;
    i = i +1;
    ln = fgetl(fileHandle);
end

fclose(fileHandle);

for i=1:numel(datasets)
    imageInPath = fullfile(root,[datasets{i} '.tif']);
    imInfo = imfinfo(imageInPath);
    im = [];
    z = 1;
    disp(datasets{i});
    fprintf('Read:');
    for j=1:numel(imInfo)
        chan = mod(j,6);
        if (chan==0)
            chan = 6;
        end
        
        im(:,:,z,1,chan) = imread(imageInPath,j,'Info',imInfo);
        
        if (chan==6)
            z = z +1;
            fprintf('.');
        end
    end
    
    im = uint8(im);
    
    fprintf('\nWrite...');
    outdir = fullfile(root, datasets{i});
    if (~exist(outdir,'file'))
        mkdir(outdir);
    end
    
    tiffWriter(im,fullfile(outdir,datasets{i}));
    fprintf('\n');
end