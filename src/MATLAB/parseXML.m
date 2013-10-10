function theStruct = parseXML(filename)
% PARSEXML Convert XML file to a MATLAB structure.
global imageData;

tree = xmlread(filename);

imageData = struct(...
    'DatasetName','',...
    'NumberOfChannels',0,...
    'NumberOfFrames',0,...
    'XDimension',0,...
    'YDimension',0,...
    'ZDimension',0,...
    'XPosition',0,...
    'YPosition',0,...
    'XPixelPhysicalSize',0,...
    'YPixelPhysicalSize',0,...
    'ZPixelPhysicalSize',0);

% Recurse over child nodes. This could run into problems
% with very deeply nested trees.
theStruct = parseChildNodes(tree);
imageData.NumberOfFrames = imageData.NumberOfFrames/imageData.ZDimension;
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
    if strcmpi(theNode.getNodeName,'ChannelDescription')
        imageData.NumberOfChannels = imageData.NumberOfChannels +1;
    end
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
    
    if strcmpi(theNode.getNodeName,'DimensionDescription')
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
                        imageData.XDimension = str2double(attributes(i).Value);
                    case 2
                        imageData.YDimension = str2double(attributes(i).Value);
                    case 3
                        imageData.ZDimension = str2double(attributes(i).Value);
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