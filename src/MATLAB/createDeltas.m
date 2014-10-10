function imageDatasets = createDeltas(imageDatasets,chan)

n = length(imageDatasets);

edges = struct(...
    'normCovar',{},...
    'deltaX',{},...
    'deltaY',{},...
    'deltaZ',{},...
    'overlapSize',{},...
    'nodeIdx1',{},...
    'nodeIdx2',{});

edges(n*(n-1)/2).normCovar = -inf;
W = ones(1,n*(n-1)/2) * -inf;
nodes1 = zeros(1,n*(n-1)/2);
nodes2 = zeros(1,n*(n-1)/2);

makeGraph = tic;
c = 1;
for i=1:n
    static = tic;
    for j=i+1:n
        [imageDataset1,~] = readMetaData(fullfile(imageDatasets(i).imageDir,[imageDatasets(i).DatasetName,'.txt']));
        [imageDataset2,~] = readMetaData(fullfile(imageDatasets(j).imageDir,[imageDatasets(j).DatasetName,'.txt']));
        [image1ROI,image2ROI] = calculateOverlap(imageDataset1,imageDataset2);
        
        if (any(image1ROI<1) || any(image2ROI<1)), continue, end
        
        [im1,imageDataset1] = tiffReader([],[],[],[],imageDatasets(i).imageDir);
        [im2,imageDataset2] = tiffReader([],[],[],[],imageDatasets(j).imageDir);        
        [deltaX,deltaY,deltaZ,normCovar,overlapSize]...
            = registerTwoImages(im1,imageDataset1,im2,imageDataset2,chan,10);
        edges(c).normCovar = normCovar;
        edges(c).deltaX = deltaX;
        edges(c).deltaY = deltaY;
        edges(c).deltaZ = deltaZ;
        edges(c).overlapSize = overlapSize;
        edges(c).nodeIdx1 = i;
        edges(c).nodeIdx2 = j;
        
        nodes1(c) = i;
        nodes2(c) = j;
        W(c) = -normCovar;
        c = c+1;
    end
    fprintf('%s took:%4.3f sec\n',imageDatasets(i).DatasetName,toc(static));
end
fprintf('Graph took:%5.3f sec\n',toc(makeGraph));

idx = find(nodes1~=0);

nodes1 = nodes1(idx);
nodes2 = nodes2(idx);
W = W(idx);
DG = sparse(nodes1,nodes2,W,n,n);
UG = tril(DG + DG');
view(biograph(UG,[],'ShowArrows','off','ShowWeights','on'));

W = W - min(W(:)) + 0.001;

DG = sparse(nodes1,nodes2,W,n,n);
UG = tril(DG + DG');
[ST,pred] = graphminspantree(UG);
view(biograph(ST,[],'ShowArrows','off','ShowWeights','on'));

edges = applyParentDeltas(ST,0,1,0,0,0,edges);

for i=1:n
    parentNode = pred(i);
    if (parentNode==0)
        deltaX = 0;
        deltaY = 0;
        deltaZ = 0;
        maxCorr = 0.0;
        parent = imageDatasets(i).DatasetName;
    else
        if (parentNode<i)
            edgeIdx = find([edges(:).nodeIdx1]==parentNode & [edges(:).nodeIdx2]==i);
        else
            edgeIdx = find([edges(:).nodeIdx2]==parentNode & [edges(:).nodeIdx1]==i);
        end
        deltaX = edges(edgeIdx).deltaX;
        deltaY = edges(edgeIdx).deltaY;
        deltaZ = edges(edgeIdx).deltaZ;
        maxCorr = edges(edgeIdx).normCovar;
        parent = imageDatasets(parentNode).DatasetName;
    end
    f = fopen(fullfile(imageDatasets(i).imageDir,'..',[imageDatasets(i).DatasetName,'_corrResults.txt']),'w');
    fprintf(f,'deltaX:%d\n',deltaX);
    fprintf(f,'deltaY:%d\n',deltaY);
    fprintf(f,'deltaZ:%d\n',deltaZ);
    fprintf(f,'MaxCorr:%f\n',maxCorr);
    fprintf(f,'Parent:%s\n',parent);
    fclose(f);
end
end