function MakeSVZMask()

processing = true;
previousDir = [];

while (processing)
    [fileName,dirName,~] = uigetfile([previousDir '.tif'],'Select MIP');
    previousDir = dirName;
    
    if (any(fileName==0) || any(dirName==0))
        break
    end
    
    [~,name,~] = fileparts(fileName);
    
    specChan = false;
    if (strcmpi(name(end-2:end),'_3c'))
        specChan = true;
        name = name(1:end-3);
    end
    
    im = imread(fullfile(dirName,fileName));  
    
    rspnce = 'yes';
    if (exist(fullfile(dirName,[name,'_RGB_mask.tif']),'file'))
        rspnce = questdlg('Would you like to redraw?','Redo ROI','Yes','No','No');
        if (strcmpi('yes',rspnce))
            bw = roipoly(im);
            if (isempty(bw))
                continue
            end
        else
            bw = imread(fullfile(dirName,[name,'_RGB_mask.tif']));
        end
    else
        bw = roipoly(im);
    end
    
    grayIm = rgb2gray(im);
    grayIm = repmat(grayIm,1,1,3);
    bwMask = repmat(bw,1,1,3);  
    
    bound = bwperim(bw);
    se = strel('disk',5,4);
    bound = imdilate(bound,se);
    boundInd = find(bound);
    
    im(~bwMask) = grayIm(~bwMask);
    im(boundInd) = 255;
    im(boundInd+numel(bw)) = 255;
    im(boundInd+numel(bw)*2) = 0;
    
    suffix = [];
    if (specChan)
        suffix = '_3c';
    end
    
    if (strcmpi('yes',rspnce))
        imwrite(bw,fullfile(dirName,[name,'_mask',suffix,'.tif']),'Compression','lzw','Description','SVZ Mask');
    end
    
    imwrite(im,fullfile(dirName,[name,'_maskRGB',suffix,'.tif']),'Compression','lzw','Description','SVZ Mask');
    
    close all
end

end
