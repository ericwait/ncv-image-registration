function edges = applyParentDeltas(minSpanTree,parentNode,curNode,deltaXoffset,deltaYoffset,deltaZoffset,edges)

children = find(minSpanTree(:,curNode));

if (parentNode~=0)
    if (parentNode<curNode)
        edgeIdx = find([edges(:).nodeIdx1]==parentNode & [edges(:).nodeIdx2]==curNode);
        sng = 1;
    else
        edgeIdx = find([edges(:).nodeIdx2]==parentNode & [edges(:).nodeIdx1]==curNode);
        sng = -1;
    end
    edges(edgeIdx).deltaX = sng*edges(edgeIdx).deltaX + deltaXoffset;
    edges(edgeIdx).deltaY = sng*edges(edgeIdx).deltaY + deltaYoffset;
    edges(edgeIdx).deltaZ = sng*edges(edgeIdx).deltaZ + deltaZoffset;
    
    deltaXoffset = edges(edgeIdx).deltaX;
    deltaYoffset = edges(edgeIdx).deltaY;
    deltaZoffset = edges(edgeIdx).deltaZ;
end

for i=1:length(children)
    edges = applyParentDeltas(minSpanTree,curNode,children(i),deltaXoffset,deltaYoffset,deltaZoffset,edges);
end
end