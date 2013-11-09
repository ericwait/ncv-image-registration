function makeZoomImages(center,path,datasetName)
global imageData
maxChannel = 6;
c = 1;
xRadius = 1023;
yRadius = 575;
while(c<=maxChannel)
    for reduc=1:5
        if (reduc==4)
            continue;
        end
        if (reduc==1)
            imagesPath = path;
        else
            imagesPath = fullfile(path,['x' num2str(reduc)]);
        end
        
        im = tiffReader('uint8',c,[],[],imagesPath,[datasetName '.txt']);
        maxChannel = imageData.NumberOfChannels;
        rowMin = round(center(1)/reduc)-yRadius;
        rowMax = round(center(1)/reduc)+yRadius;
        colMin = round(center(2)/reduc)-xRadius;
        colMax = round(center(2)/reduc)+xRadius;
        
        if (rowMax>size(im,1))
            rowMax = size(im,1);
            rowMin = max(1,size(im,1)-yRadius*2);
        elseif (rowMin<1)
            rowMin=1;
            rowMax = min(size(im,1),yRadius*2);
        end
        if (colMax>size(im,2))
            colMax = size(im,2);
            colMin = max(1,colMax-xRadius*2);
        elseif (colMin<1)
            colMin=1;
            colMax = min(size(im,2),colMin+xRadius*2);
        end
        
        imROI = im(rowMin:rowMax,colMin:colMax,:);
        
        if (~exist(fullfile(path,'movie',['x' num2str(reduc)]),'file'))
            mkdir(fullfile(path,'movie',['x' num2str(reduc)]));
        end
        
        imageData.xDim = colMax-colMin+1;
        imageData.yDim = rowMax-rowMin+1;
        createMetadata(fullfile(path,'movie',['x' num2str(reduc)]),datasetName,imageData);
        
        tiffWriter(imROI,[fullfile(path,'movie',['x' num2str(reduc)]) '\' datasetName],c);
        
        clear im;
        clear imROI;
    end
    
    c = c +1;
end
