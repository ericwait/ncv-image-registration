function tiffWriter(image,prefix,chanList,timeList,zList)
sizes = size(image);
numDim = length(sizes);

if (~exist('chanList','var') || isempty(chanList))
    if (numDim>4)
        chanList = 1:sizes(5);
    else
        chanList = 1;
    end
end
if (~exist('timeList','var') || isempty(timeList))
    if (numDim>3)
        timeList = 1:sizes(4);
    else
        timeList = 1;
    end
end
if (~exist('zList','var') || isempty(zList))
    if (numDim>2)
        zList = 1:sizes(3);
    else
        zList = 1;
    end
end


for c=1:length(chanList)
    for t=1:length(timeList)
        for z=1:length(zList)
            fileName = sprintf('%s_c%d_t%04d_z%04d.tif',prefix,chanList(c),timeList(t),zList(z));
            imwrite(image(:,:,z,t,c),fileName,'tif','Compression','lzw');
        end
    end
end

end

