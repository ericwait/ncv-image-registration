function imageDatasets = createDeltas(imageDatasets,visualize)
minOverlap = 25;
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

edgeEmpty = struct(...
    'i',0,...
    'j',0,...
    'normCovar',-inf,...
    'deltaX',0,...
    'deltaY',0,...
    'deltaZ',0,...
    'overlapSize',0);

edges = edgeEmpty;

edges(n,n).normCovar = -inf;

if (strcmp(reRun,'Rerun'))
    poolobj = gcp();
    
    [~, sysName] = system('hostname');
    logFile = fullfile(logDir,sprintf('graphBuild_%s.txt',sysName(1:end-1)));
    
    c = clock;
    fHand = fopen(logFile,'wt');
    fprintf(fHand,'\n%d-%02d-%02d %02d:%02d Workers:%d\n',c(1),c(2),c(3),c(4),c(5),poolobj.NumWorkers);
    fclose(fHand);
    
    e = 0;
    makeGraph = tic;

    dirs = cellstr({imageDatasets(:).imageDir});
    names = cellstr({imageDatasets(:).DatasetName});
    
    spmd        
        createEdgeData(dirs,names, logFile,minOverlap,maxSearchSize,visualize);
    end
    
    delete(poolobj);
    
    dirList = dir(fullfile(dirs(1),'*.mat'));
    for k=1:length(dirList)
        ed = load(fullfile(dirs(1),dirList(k).name));
        edges(ed.i,ed.j) = ed;
        e = e+1;
    end
    
    tm = toc(makeGraph);
    
    fHand = fopen(logFile,'at');
    fprintf(fHand,'Graph creation took: %s, per edge %06.3f sec, per worker %f sec\n',printTime(tm),tm/e,poolobj.NumWorkers);
    fclose(fHand);
    
    save(fullfile(logDir,'graphEdges.mat'),'edges');
else
    load(fullfile(logDir,'graphEdges.mat'));
end

bEmpty = arrayfun(@(x)(isempty(x.normCovar)),edges);
ncv = -inf*ones(n,n);
ncv(~bEmpty) = [edges.normCovar];

[nodes1,nodes2] = find(~isinf(ncv));
ncvGraph = sparse(nodes1,nodes2,ncv(~isinf(ncv)),n,n);
minW = max(ncv(~isinf(ncv))) - ncv(~isinf(ncv)) + 0.001;
weightGraph = sparse(nodes1,nodes2,minW,n,n);

if (visualize)
    ugOrg = tril(ncvGraph + ncvGraph');
    view(biograph(ugOrg,[],'ShowArrows','off','ShowWeights','on'));
end

ugW = tril(weightGraph + weightGraph');
[minSpanTree,pred] = graphminspantree(ugW);
ncvSem = ncvGraph - ncvGraph';

if (visualize)
    nzEdge = minSpanTree~=0;
    minSpanTree(nzEdge) = ncvSem(nzEdge);
    view(biograph(minSpanTree,[],'ShowArrows','off','ShowWeights','on'));
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
            edgeIdx = sub2ind(size(edges),parentNode,i);
            sgn = 1;
        else
            edgeIdx = sub2ind(size(edges),i,parentNode);
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
