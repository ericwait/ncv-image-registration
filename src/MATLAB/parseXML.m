function theStruct = parseXML(filename)
% PARSEXML Convert XML file to a MATLAB structure.
global imageData;

tree = xmlread(filename);

if isempty(imageData)
    imageData = struct(...
        'DatasetName','',...
        'NumberOfChannels',0,...
        'ChannelColors',{},...
        'NumberOfFrames',0,...
        'xDim',0,...
        'yDim',0,...
        'zDim',0,...
        'XPosition',0,...
        'YPosition',0,...
        'XPixelPhysicalSize',0,...
        'YPixelPhysicalSize',0,...
        'ZPixelPhysicalSize',0,...
        'XDistanceUnits','',...
        'YDistanceUnits','',...
        'ZDistanceUnits','',...
        'XLength',0,...
        'YLength',0,...
        'ZLength',0);
end

if (isempty(imageData))
    imageData(1).NumberOfChannels = 0;
end

oldNumChannels = imageData.NumberOfChannels;
oldColors = imageData.ChannelColors;

imageData.NumberOfChannels = 0;
imageData.ChannelColors = {};

% Recurse over child nodes. This could run into problems
% with very deeply nested trees.
theStruct = parseChildNodes(tree);
if (~isempty(imageData.NumberOfFrames) && ~isempty(imageData.zDim))
    imageData.NumberOfFrames = imageData.NumberOfFrames/imageData.zDim;
end
if (length(imageData.NumberOfChannels)<length(oldNumChannels))
    imageData.NumberOfChannels = oldNumChannels;
    imageData.ChannelColors = oldColors;
end

end


% ----- Subfunction PARSECHILDNODES -----
function children = parseChildNodes(theNode)
global imageData;
% Recurse over node children.
children = [];
if theNode.hasChildNodes
    childNodes = theNode.getChildNodes;
    numChildNodes = childNodes.getLength;
    allocCell = cell(1, numChildNodes);
    
    children = struct(             ...
        'Name', allocCell, 'Attributes', allocCell,    ...
        'Data', allocCell, 'Children', allocCell);
    
    if strcmpi(theNode.getNodeName,'FrameCount') && theNode.hasChildNodes
        item = childNodes.item(0);
        data = char(item.getData);
        ind = strfind(data,',');
        if ~isempty(ind)
            ind2 = strfind(data(ind+2:end),' ');
            imageData.NumberOfFrames = str2double(data(ind+2:ind+ind2));
        end
    elseif strcmpi(theNode.getNodeName,'Name') && theNode.hasChildNodes
        item = childNodes.item(0);
        imageData.DatasetName = char(item.getData);
    end
    
    for count = 1:numChildNodes
        theChild = childNodes.item(count-1);
        children(count) = makeStructFromNode(theChild);
    end
end
end

% ----- Subfunction MAKESTRUCTFROMNODE -----
function nodeStruct = makeStructFromNode(theNode)
% Create structure of node info.

nodeStruct = struct(                        ...
    'Name', char(theNode.getNodeName),       ...
    'Attributes', parseAttributes(theNode),  ...
    'Data', '',                              ...
    'Children', parseChildNodes(theNode));

if any(strcmp(methods(theNode), 'getData'))
    nodeStruct.Data = char(theNode.getData);
else
    nodeStruct.Data = '';
end
end

% ----- Subfunction PARSEATTRIBUTES -----
function attributes = parseAttributes(theNode)
global imageData;
% Create attributes structure.

attributes = [];
if theNode.hasAttributes
    theAttributes = theNode.getAttributes;
    numAttributes = theAttributes.getLength;
    allocCell = cell(1, numAttributes);
    attributes = struct('Name', allocCell, 'Value', ...
        allocCell);
    
    for count = 1:numAttributes
        attrib = theAttributes.item(count-1);
        attributes(count).Name = char(attrib.getName);
        attributes(count).Value = char(attrib.getValue);
    end
    
    if strcmpi(theNode.getNodeName,'ChannelDescription')
        imageData.NumberOfChannels = imageData.NumberOfChannels +1;
        for i = 1:numAttributes
            if strcmpi(attributes(i).Name,'LUTName')
                imageData.ChannelColors{end+1} = attributes(i).Value;
            end
        end
    elseif strcmpi(theNode.getNodeName,'DimensionDescription')
        dim = 0;
        for i = 1:numAttributes
            if strcmpi(attributes(i).Name,'DimID')
                if strcmpi(attributes(i).Value,'X')
                    dim = 1;
                elseif strcmpi(attributes(i).Value,'Y')
                    dim = 2;
                elseif strcmpi(attributes(i).Value,'Z')
                    dim = 3;
                end
            end
        end
        for i = 1:numAttributes
            if strcmpi(attributes(i).Name,'NumberOfElements')
                switch dim
                    case 1
                        imageData.xDim = str2double(attributes(i).Value);
                    case 2
                        imageData.yDim = str2double(attributes(i).Value);
                    case 3
                        imageData.zDim = str2double(attributes(i).Value);
                end
            elseif strcmpi(attributes(i).Name,'Voxel')
                switch dim
                    case 1
                        imageData.XPixelPhysicalSize = str2double(attributes(i).Value);
                    case 2
                        imageData.YPixelPhysicalSize = str2double(attributes(i).Value);
                    case 3
                        imageData.ZPixelPhysicalSize = str2double(attributes(i).Value);
                end
            elseif strcmpi(attributes(i).Name,'Unit')
                switch dim
                    case 1
                        imageData.XDistanceUnits = attributes(i).Value;
                    case 2
                        imageData.YDistanceUnits = attributes(i).Value;
                    case 3
                        imageData.ZDistanceUnits = attributes(i).Value;
                end
            elseif strcmpi(attributes(i).Name,'Length')
                switch dim
                    case 1
                        imageData.XLength = str2double(attributes(i).Value);
                    case 2
                        imageData.YLength = str2double(attributes(i).Value);
                    case 3
                        imageData.ZLength = str2double(attributes(i).Value);
                end
            end
        end
    elseif strcmpi(theNode.getNodeName,'FilterSettingRecord')
        dim = 0;
        for i = 1:numAttributes
            if strcmpi(attributes(i).Value,'XPos')
                dim = 1;
            elseif strcmpi(attributes(i).Value,'YPos')
                dim = 2;
            end
        end
        for i = 1:numAttributes
            if strcmpi(attributes(i).Name,'Variant');
                switch dim
                    case 1
                        imageData.XPosition = str2double(attributes(i).Value);
                    case 2
                        imageData.YPosition = str2double(attributes(i).Value);
                end
            end
        end
    end
end
end