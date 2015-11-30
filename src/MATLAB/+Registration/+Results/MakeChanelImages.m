function MakeChanelImages()

load('montages.mat');

for i=1:length(montages)
    chans = montages(i).chanList(2:end);
    [im,imD] = MicroscopeData.Reader(montages(i).filePath,[],chans);
    colors = MicroscopeData.GetChannelColors(imD);
    im = ImUtils.ConvertType(im,'uint8',true);
    imFinal = ImUtils.ThreeD.ColorMIP(im,colors(chans));   
    imwrite(imFinal,ChanNumbers(montages(i).filePath,chans),'compression','lzw');
    for c=1:length(chans)
        imFinal = ImUtils.ThreeD.ColorMIP(im(:,:,:,c),colors(chans(c)));
        imwrite(imFinal,ChanNumbers(montages(i).filePath,chans(c)),'compression','lzw');
    end
end
end

function path = ChanNumbers(curPath,chanList)
path = [curPath,'_chans'];
for i=1:length(chanList)
    path = [path,num2str(chanList(i))];
end
path = [path,'.tif'];
end