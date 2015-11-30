for i=1:length(montages)
    fullStart = tic;
    readStart = tic;
    [pathstr,name,ext] = fileparts(montages(i).filePath);
    imData = MicroscopeData.ReadMetadata(montages(i).filePath);
    
    if (~exist(fullfile(pathstr,[imData.DatasetName,'_seg.json']),'file'))
        imBW = false(imData.YDimension,imData.XDimension,imData.ZDimension,length(montages(i).chanList));
    else
        imBW = MicroscopeData.Reader(fullfile(pathstr,[imData.DatasetName,'_seg.json']),[],[],[],'logical');
    end
    
    disp(fullfile(pathstr,['_',imData.DatasetName,'_mask.tif']));
    imMask = imread(fullfile(pathstr,['_',imData.DatasetName,'_mask.tif']));
    imMask = repmat(imMask,1,1,imData.ZDimension);
    
    processStart = tic;
    %for c=1:length(montages(i).chanList)
    for c=2
        im = MicroscopeData.Reader(montages(i).filePath,[],montages(i).chanList(c));
        if (c==2)
            imG = Cuda.CudaMex('GaussianFilter',im,[25,25,10]);
            im = im - 2*imG;
            clear imG
            im = Denoise.FluorescentBGRemoval(im);
            imBW(:,:,:,c) = im > 0;
        else
            if (c==3)
                im = Denoise.FluorescentBGRemoval(im,[],10^-8);
            else
                im = Denoise.FluorescentBGRemoval(im);
            end
            im(~imMask) = 0;
            th = graythresh(im(im>0));
            [numBits,minVal,maxVal] = Utils.GetClassBits(im);
            imBW(:,:,:,c) = im>=(th*maxVal);
        end
    end
    processTime = toc(processStart);
    ImUtils.ThreeD.ShowMaxImage(imBW,1);
    
    writeStart = tic;
    imData.DatasetName = [imData.DatasetName,'_seg'];
    imData.NumberOfChannels = length(montages(i).chanList);
    MicroscopeData.Writer(imBW,pathstr,imData);
    writeTime = toc(writeStart);
    fullTime = toc(fullStart);
    
    fprintf('TotalTime:%s, process:%s, write:%s\n',...
        Utils.PrintTime(fullTime),Utils.PrintTime(processTime),Utils.PrintTime(writeTime));
end

clear imMask
clear imBW
clear im
