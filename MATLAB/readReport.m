reportDir = uigetdir();
fileList = dir(fullfile(reportDir,'*_report.txt'));
if ~isempty(fileList)
    fprintf('%d:',size(fileList,1));
    for i=1:size(fileList,1)
        fprintf('%d,',i);
        reportFile = fileList(i).name;
        
        aviName = sprintf('%s',fullfile(reportDir,reportFile));
        aviName = aviName(1:end-4);
        
%         if (exist([aviName '.avi'],'file'))
%             continue;
%         end

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
        
        corrMap = zeros(sizeX,sizeY,sizeZ);
        staticSigs = zeros(sizeX,sizeY,sizeZ);
        overlapSigs = zeros(sizeX,sizeY,sizeZ);
        numVoxels = zeros(sizeX,sizeY,sizeZ);
        
        for i=1:length(cMap{1})
            corrMap(cMap{1}(i)-minX+1,cMap{2}(i)-minY+1,cMap{3}(i)-minZ+1) = cMap{4}(i);
            staticSigs(cMap{1}(i)-minX+1,cMap{2}(i)-minY+1,cMap{3}(i)-minZ+1) = cMap{5}(i);
            overlapSigs(cMap{1}(i)-minX+1,cMap{2}(i)-minY+1,cMap{3}(i)-minZ+1) = cMap{6}(i);
            numVoxels(cMap{1}(i)-minX+1,cMap{2}(i)-minY+1,cMap{3}(i)-minZ+1) = cMap{7}(i);
        end
        
        corMax = max(corrMap(:));
        corMin = min(corrMap(:));
        maxIdx = find(corrMap==corMax);
        [mXcor mYcor mZcor] = ind2sub(size(corrMap),maxIdx);
        
        f = figure('Renderer','zbuffer','Position',[130 130 1080 1080]);
        surf(corrMap(:,:,1));
        axis tight;
        set(gca,'NextPlot','replaceChildren');
        
        writer = VideoWriter(aviName,'Uncompressed AVI');
        writer.FrameRate = 15;
        %writer.Quality = 100;
        
        open(writer);
        
        for z=1:size(corrMap,3)
            h = surf(corrMap(:,:,z));
            xlim([0 100]);
            ylim([0 100]);
            zlim([-0.2 1.0]);
            caxis([-0.2 1.0]);
            title({reportFile(1:end-4);sprintf('Z=%d',z)},'Interpreter','none');
            xlabel('Delta X');
            ylabel('Delta Y');
            zlabel('Correlation');
            
            j = find(mZcor==z);
            if ~isempty(j)
                hold on
                plot3(mYcor(j),mXcor(j),corrMap(mXcor(j),mYcor(j),z),'o','MarkerFaceColor','b','MarkerEdgeColor','w','MarkerSize',10);
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
