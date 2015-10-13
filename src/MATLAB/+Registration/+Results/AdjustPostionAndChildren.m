function adjustPostionAndChildren(idx, xDelta, yDelta, zDelta)
global imageDatasets

imageDatasets(idx).xMinPos = imageDatasets(idx).xMinPos + xDelta;
imageDatasets(idx).yMinPos = imageDatasets(idx).yMinPos + yDelta;
imageDatasets(idx).zMinPos = imageDatasets(idx).zMinPos + zDelta;

imageDatasets(idx).xMaxPos = imageDatasets(idx).xMaxPos + xDelta;
imageDatasets(idx).yMaxPos = imageDatasets(idx).yMaxPos + yDelta;
imageDatasets(idx).zMaxPos = imageDatasets(idx).zMaxPos + zDelta;

for i=1:length(imageDatasets(idx).Child)
    adjustPostionAndChildren(imageDatasets(idx).Child(i),xDelta,yDelta,zDelta);
end

end