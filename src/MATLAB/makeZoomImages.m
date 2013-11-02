function makeZoomImages(center,path,datasetName)
global imageData
maxChannel = 2;
c = 1;
xRadius = 1023;
yRadius = 360;
while(c<=maxChannel)
    for reduc=1:7
        if (reduc==1)
            imagesPath = path;
        else
            imagesPath = fullfile(path,['x' num2str(reduc)]);
        end
        
        im = tiffReader('uint8',c,[],[],imagesPath,[datasetName '.txt']);
        maxChannel = imageData.NumberOfChannels;
        rowMin = max(1,round(center(1)/reduc)-yRadius);
        rowMax = min(size(im,1),round(center(1)/reduc)+yRadius);
        colMin = max(1,round(center(2)/reduc)-xRadius);
        colMax = min(size(im,2),round(center(2)/reduc)+xRadius);
        
        if (rowMax-rowMin<yRadius*2)
            if (rowMax==size(im,1))
                rowMin = max(1,rowMax-yRadius*2);
            else
                rowMax = min(size(im,1),rowMin+yRadius*2);
            end
        end
        if (colMax-colMin<xRadius*2)
            if (colMax==size(im,2))
                colMin = max(1,colMax-xRadius*2);
            else
                colMax = min(size(im,2),colMin+xRadius*2);
            end
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
