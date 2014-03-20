function readXMLmetadata( root )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global imageData

fileList = dir( fullfile([root '\'],'*.xml'));
for i=1:length(fileList)
    parseXML([root '\' fileList(i).name]);
end

createMetadata(fullfile(root,'..'),imageData);

clear imageData;

