function imageDatasets = createDeltas(imageDatasets,chan,visualize)
minOverlap = 50;
maxSearchSize = 100;
n = length(imageDatasets);

logDir = fullfile(imageDatasets(1).imageDir,'..','_GraphLog');
if (~exist(logDir,'dir'))
    mkdir(logDir);
end

if (exist(fullfile(logDir,'graphEdges.mat'),'file'))
    reRun = questdlg('Rerun registration or use old graph?','Rerun Registration','Rerun','Old','Old');
    if (reRun==0)
        return
    end
else
    reRun = 'Rerun';
end

edges = struct(...
    'normCovar',-inf,...
    'deltaX',0,...
    'deltaY',0,...
    'deltaZ',0,...
    'overlapSize',0);

edges(n,n).normCovar = -inf;

if (strcmp(reRun,'Rerun'))
    poolobj = gcp();
    
    [~, sysName] = system('hostname');
    logFile = fullfile(logDir,sprintf('graphBuild_%s.txt',sysName(1:end-1)));
    
    c = clock;
    fHand = fopen(logFile,'wt');
    fprintf(fHand,'%d-%02d-%02d %02d:%02d Workers:%d\n',c(1),c(2),c(3),c(4),c(5),poolobj.NumWorkers);
    fclose(fHand);
    
    e = 0;
    makeGraph = tic;
    for i=1:n
        static = tic;
        [im1,imageDataset1] = tiffReader([],chan,[],[],imageDatasets(i).imageDir,1);
        
        for j=i+1:n
            [im2,imageDataset2] = tiffReader([],chan,[],[],imageDatasets(j).imageDir,1);
            
            [~,~,minXdist,minYdist] = calculateOverlap(imageDataset1,imageDataset2);
            
            if (minXdist>maxSearchSize-minOverlap || minYdist>maxSearchSize-minOverlap)
                ed.normCovar = -inf;
                ed.deltaX = 0;
                ed.deltaY = 0;
                ed.deltaZ = 0;
                ed.overlapSize = 0;
                idx = sub2ind(size(edges),i,j);
                edges(idx) = ed;
            else
                [deltaX,deltaY,deltaZ,normCovar,overlapSize] = registerTwoImages(im1,imageDataset1,im2,imageDataset2,...
                    minOverlap,maxSearchSize,logFile,visualize,visualize);
                
                ed.normCovar = normCovar;
                ed.deltaX = deltaX;
                ed.deltaY = deltaY;
                ed.deltaZ = deltaZ;
                ed.overlapSize = overlapSize;
                idx = sub2ind(size(edges),i,j);
                edges(idx) = ed;
                e = e+1;
                fprintf(1,'.');
            end
        end
        clear('im1');
        clear('im2');
        tm = toc(static);
        fHand = fopen(logFile,'at');
        fprintf(fHand,'%s took %s\n',imageDatasets(i).DatasetName,printTime(tm));
        fprintf(1,'\n%s took %s\n',imageDatasets(i).DatasetName,printTime(tm));
        fclose(fHand);
    end
    
    tm = toc(makeGraph);

    fHand = fopen(logFile,'at');
    fprintf(fHand,'Graph creation took: %s, per edge %06.3f sec\n',printTime(tm),tm/e);
    fclose(fHand);
    delete(poolobj);
    
    save(fullfile(logDir,'graphEdges.mat'),'edges');
else
    load(fullfile(logDir,'graphEdges.mat'));
end

nodes1 = zeros(1,length(edges));
nodes2 = zeros(1,length(edges));
W = zeros(1,length(edges));

for i=1:length(edges)
    if (~isempty(edges(i).nodeIdx1))
        nodes1(i)=edges(i).nodeIdx1;
        nodes2(i)=edges(i).nodeIdx2;
        W(i)=edges(i).normCovar;
    end
end

idx = find(nodes1~=0);
nodes1 = nodes1(idx);
nodes2 = nodes2(idx);
W = W(idx);

DG = sparse(nodes1,nodes2,W,n,n);
UG = tril(DG + DG');

if (visualize)
    view(biograph(UG,[],'ShowArrows','off','ShowWeights','on'));
end

minW = max(W(:)) - W + 0.001;
DG = sparse(nodes1,nodes2,minW,n,n);
UG = tril(DG + DG');
[~,pred] = graphminspantree(UG);

if (visualize)
    minTree = [pred;1:length(pred)].';
    minTree = minTree(minTree(:,1)~=0,:);
    DG = sparse(nodes1,nodes2,W,n,n);
    UG = (DG + DG');
    idx = sub2ind(size(UG),minTree(:,1),minTree(:,2));
    ST = sparse(minTree(:,1),minTree(:,2),UG(idx),n,n);
    
    view(biograph(ST,[],'ShowArrows','off','ShowWeights','on'));
end

for i=1:n
    parentNode = pred(i);
    if (parentNode==0)
        deltaX = 0;
        deltaY = 0;
        deltaZ = 0;
        ncv = 0.0;
        parent = imageDatasets(i).DatasetName;
    else
        if (parentNode<i)
            edgeIdx = find([edges(:).nodeIdx1]==parentNode & [edges(:).nodeIdx2]==i);
            sgn = 1;
        else
            edgeIdx = find([edges(:).nodeIdx2]==parentNode & [edges(:).nodeIdx1]==i);
            sgn = -1;
        end
        deltaX = sgn*edges(edgeIdx).deltaX;
        deltaY = sgn*edges(edgeIdx).deltaY;
        deltaZ = sgn*edges(edgeIdx).deltaZ;
        ncv = edges(edgeIdx).normCovar;
        parent = imageDatasets(parentNode).DatasetName;
    end
    
    f = fopen(fullfile(imageDatasets(i).imageDir,'..',[imageDatasets(i).DatasetName,'_corrResults.txt']),'w');
    fprintf(f,'deltaX:%d\n',deltaX);
    fprintf(f,'deltaY:%d\n',deltaY);
    fprintf(f,'deltaZ:%d\n',deltaZ);
    fprintf(f,'NCV:%f\n',ncv);
    fprintf(f,'Parent:%s\n',parent);
    fclose(f);
end
end
