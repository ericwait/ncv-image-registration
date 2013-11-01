function readXMLmetadata( root, datasetName )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global imageData;

fileList = dir( fullfile([root '\'],'*.xml'));
for i=1:length(fileList)
    parseXML([root '\' fileList(i).name]);
    createMetadata(root,datasetName,imageData);
end

