reportDir = uigetdir();
fileList = dir(fullfile(reportDir,'*_report.txt'));

stepSize = 3;
if ~isempty(fileList)
    fprintf('%d:',size(fileList,1));
    for i=1:size(fileList,1)
        fprintf('%d,',i);
        reportFile = fileList(i).name;
        
        aviName = sprintf('%s',fullfile(reportDir,reportFile));
        aviName = aviName(1:end-4);
        
        if (exist([aviName '.avi'],'file'))
            continue;
        end

        fid = fopen(fullfile(reportDir,reportFile),'r');
        
        cMap = [];
        while ~feof(fid)
            C = textscan(fid,'(%d,%d,%d):%f,%f,%f,%d');
            cMap = [cMap;C];
        end
        
        fclose(fid);
        
        minX = min(cMap{1});
        maxX = max(cMap{1});
        minY = min(cMap{2});
        maxY = max(cMap{2});
        minZ = min(cMap{3});
        maxZ = max(cMap{3});
        
        sizeX = maxX - minX +1;
        sizeY = maxY - minY +1;
        sizeZ = maxZ - minZ +1;
        
        if (sizeX<stepSize+1 || sizeY<stepSize+1)
            continue;
        end
        
        corrMap = zeros(sizeX,sizeY,sizeZ);
        staticSigs = zeros(sizeX,sizeY,sizeZ);
        overlapSigs = zeros(sizeX,sizeY,sizeZ);
        numVoxels = zeros(sizeX,sizeY,sizeZ);
        
        coords = double([cMap{1}(:) cMap{2}(:) cMap{3}(:)]);
        dif = double([(minX-1) (minY-1) (minZ-1)]);
        coords = bsxfun(@minus,coords,dif);
        
        for i=1:length(cMap{1})
            corrMap(coords(i,1),coords(i,2),coords(i,3)) = double(cMap{4}(i));
            staticSigs(coords(i,1),coords(i,2),coords(i,3)) = double(cMap{5}(i));
            overlapSigs(coords(i,1),coords(i,2),coords(i,3)) = double(cMap{6}(i));
            numVoxels(coords(i,1),coords(i,2),coords(i,3)) = double(cMap{7}(i));
        end
        
        corMax = max(corrMap(:));
        corMin = min(corrMap(:));
        maxStaticSig = max(staticSigs(:));
        maxOverlapSig = max(overlapSigs(:));
        
        staticCorr = corrMap.*staticSigs;
        overlapCorr = corrMap.*overlapSigs;
        
        maxIdx = find(corrMap==corMax);
        [mXcor mYcor mZcor] = ind2sub(size(corrMap),maxIdx);
        
        %% render
        f = figure('Renderer','zbuffer','Position',[130 130 1920 1080]);
        %Correlation Static
        subplot(1,3,2,'Position',[.35 .03 .3 .9]);
        surf(minY:3:maxY,minX:3:maxX,staticCorr(1:3:end,1:3:end,1));
        title({reportFile(1:end-4);sprintf('Z=%d',z)},'Interpreter','none');
        xlim([-100 100]);
        ylim([-100 100]);
        zlim([-0.2 1.0].*maxStaticSig);
        caxis([-0.2 1.0].*maxStaticSig);
        xlabel('Delta X');
        ylabel('Delta Y');
        %zlabel('Static Sig*Correlation');
        set(gca,'NextPlot','replaceChildren');
        
        %Correlation Overlap
        subplot(1,3,3,'Position',[.68 .03 .3 .9]);
        surf(minY:3:maxY,minX:3:maxX,overlapCorr(1:3:end,1:3:end,1));
        xlim([-100 100]);
        ylim([-100 100]);
        zlim([-0.2 1.0].*maxOverlapSig);
        caxis([-0.2 1.0].*maxOverlapSig);
        xlabel('Delta X');
        ylabel('Delta Y');
        %zlabel('Overlap Sig*Correlation');
        set(gca,'NextPlot','replaceChildren');
        
        %Correlation Straight
        subplot(1,3,1,'Position',[.02 .03 .3 .9]);
        surf(minY:3:maxY,minX:3:maxX,corrMap(1:3:end,1:3:end,1));
        xlim([-100 100]);
        ylim([-100 100]);
        zlim([-0.2 1.0]);
        caxis([-0.2 1.0]);
        xlabel('Delta X');
        ylabel('Delta Y');
        %zlabel('Correlation');
        set(gca,'NextPlot','replaceChildren');
        
        writer = VideoWriter(aviName,'Uncompressed AVI');
        writer.FrameRate = 15;
        %writer.Quality = 100;
        
        open(writer);
        
        for z=1:size(corrMap,3)            
            %Correlation Static
            subplot(1,3,2,'Position',[.35 .03 .3 .9]);
            surf(minY:3:maxY,minX:3:maxX,staticCorr(1:3:end,1:3:end,z));
            title({reportFile(1:end-4);sprintf('Z=%d',z)},'Interpreter','none');
            xlim([-100 100]);
            ylim([-100 100]);
            zlim([-0.2 1.0].*maxStaticSig);
            caxis([-0.2 1.0].*maxStaticSig);
            xlabel('Delta X');
            ylabel('Delta Y');
            %zlabel('Static Sig*Correlation');
            
            %Correlation Overlap
            subplot(1,3,3,'Position',[.68 .03 .3 .9]);
            surf(minY:3:maxY,minX:3:maxX,overlapCorr(1:3:end,1:3:end,z));
            xlim([-100 100]);
            ylim([-100 100]);
            zlim([-0.2 1.0].*maxOverlapSig);
            caxis([-0.2 1.0].*maxOverlapSig);
            xlabel('Delta X');
            ylabel('Delta Y');
            %zlabel('Overlap Sig*Correlation');
            
            %Correlation Straight
            subplot(1,3,1,'Position',[.02 .03 .3 .9]);
            surf(minY:3:maxY,minX:3:maxX,corrMap(1:3:end,1:3:end,z));
            xlim([-100 100]);
            ylim([-100 100]);
            zlim([-0.2 1.0]);
            caxis([-0.2 1.0]);
            xlabel('Delta X');
            ylabel('Delta Y');
            %zlabel('Correlation');
            j = find(mZcor==z);
            if ~isempty(j)
                hold on
                plot3(mYcor(j)+minY,mXcor(j)+minX,corrMap(mXcor(j),mYcor(j),z),'o','MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',10);
                frm = getframe(gcf);
                for k=1:5
                    writeVideo(writer,frm);
                end
                hold off
            end
            
            frm = getframe(gcf);
            writeVideo(writer,frm);
        end
        
        close(writer);
        
        close(f);
    end
end

fprintf('Done\n');
