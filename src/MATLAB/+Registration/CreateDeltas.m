function imageDatasets = CreateDeltas(imageDatasets,minOverlap,maxSearchSize,unitFactor,visualize)
%% Check inputs
if (~exist('minOverlap','var') || isempty(minOverlap))
    minOverlap = 25;
end

if (~exist('maxSearchSize','var') || isempty(maxSearchSize))
    maxSearchSize = 100;
end

if (~exist('visualize','var') || isempty(visualize))
    visualize = false;
end

%% Setup a log directory
logDir = fullfile(imageDatasets(1).imageDir,'..','_GraphLog');
if (~exist(logDir,'dir'))
    mkdir(logDir);
end

% See if there are results already
if (exist(fullfile(logDir,'graphEdges.mat'),'file'))
    reRun = questdlg('Rerun registration or use old graph?','Rerun Registration','Rerun','Old','Old');
    if (reRun==0)
        return
    end
else
    reRun = 'Rerun';
end

%% Create varaiables to represent the graph
numNodes = length(imageDatasets);

edgeEmpty = struct(...
    'i',0,...
    'j',0,...
    'normCovar',-inf,...
    'deltaX',0,...
    'deltaY',0,...
    'deltaZ',0,...
    'overlapSize',0);
edges = edgeEmpty;
edges(numNodes,numNodes).normCovar = -inf;

%% Create the edge graph
if (strcmp(reRun,'Rerun'))  
    % create a log file for this computer to be tracked on the network
    [~, sysName] = system('hostname');
    logFile = fullfile(logDir,sprintf('graphBuild_%s.txt',sysName(1:end-1)));
    
    % Print out the start time of this job
    c = clock;
    fHand = fopen(logFile,'wt');
    fprintf(fHand,'\n%d-%02d-%02d %02d:%02d \n',c(1),c(2),c(3),c(4),c(5));
    fclose(fHand);

    % Create structres for the parallel processes
    dirs = cellstr({imageDatasets(:).imageDir});
    names = cellstr({imageDatasets(:).DatasetName});
    
    % Start creating the edge weights in parallel
    makeGraphStartTime = tic;
    Registration.CreateEdgeData(dirs,names,logFile,minOverlap,maxSearchSize,unitFactor,visualize);
    
    % Collate the results
    e = 0;
    for i=1:length(dirs)
        dList = dir(fullfile(dirs{i},'*.mat'));
        for k=1:length(dList)
            load(fullfile(dirs{i},dList(k).name));
            if (ed.i>0 && ed.j>0)
                edges(ed.i,ed.j) = ed;
                e = e+1;
            end
            delete(fullfile(dirs{i},dList(k).name));
        end
    end
    
    tm = toc(makeGraphStartTime);
    
    % Print out the time it took 
    fHand = fopen(logFile,'at');
    fprintf(fHand,'Graph creation took: %s, per edge %06.3f sec\n',Utils.PrintTime(tm),tm/e);
    fclose(fHand);
    
    % Save the results
    save(fullfile(logDir,'graphEdges.mat'),'edges');
else
    %% Just read in the edge graph
    load(fullfile(logDir,'graphEdges.mat'));
end

%% Create a max spanning tree
bEmpty = arrayfun(@(x)(isempty(x.normCovar)),edges);
ncv = -inf*ones(numNodes,numNodes);
ncv(~bEmpty) = [edges.normCovar];

[nodes1,nodes2] = find(~isinf(ncv));
ncvGraph = sparse(nodes1,nodes2,ncv(~isinf(ncv)),numNodes,numNodes);
minW = max(ncv(~isinf(ncv))) - ncv(~isinf(ncv)) + 0.001;
weightGraph = sparse(nodes1,nodes2,minW,numNodes,numNodes);

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

%% Write out the results
for i=1:numNodes
    parentNode = pred(i);
    if (parentNode==0  || isnan(parentNode))
        deltaX = 0;
        deltaY = 0;
        deltaZ = 0;
        ncv = 0.0;
        if (i~=1)
            parent = imageDatasets(i-1).DatasetName;
        else
            parent = imageDatasets(i).DatasetName;
        end
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
    
    f = fopen(fullfile(imageDatasets(i).imageDir,[imageDatasets(i).DatasetName,'_corrResults.txt']),'w');
    fprintf(f,'deltaX:%d\n',deltaX);
    fprintf(f,'deltaY:%d\n',deltaY);
    fprintf(f,'deltaZ:%d\n',deltaZ);
    fprintf(f,'NCV:%f\n',ncv);
    fprintf(f,'Parent:%s\n',parent);
    fclose(f);
end
end
