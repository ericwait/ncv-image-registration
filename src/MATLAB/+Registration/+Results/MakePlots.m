starIdx = 1;
font = 'Consolas';
fntSize = 15;
horz = 'center';
vert = 'middle';
mrkSize = 30;

load('features.mat');

datasetNames = {};

for i=starIdx:length(montages)
    imMeta = MicroscopeData.ReadMetadata(montages(i).filePath);
    datasetNames{i-starIdx+1} = imMeta.DatasetName;
end

figure
pH = gca;
hold on

for i=starIdx:length(montages)
    x = features(i).numVox(2);
    y = features(i).numVox(3);
    plot(pH,x,y,...
        'MarkerFaceColor',montages(i).faceColor,...
        'marker',montages(i).marker,...
        'MarkerSize',mrkSize);
    text(x,y,num2str(i),...
        'parent',pH,...
        'color',xor([1,1,1],montages(i).faceColor),...
        'fontsize',fntSize,'fontname',font,...
        'horizontalalignment',horz,...
        'verticalalignment',vert);
end

title('Voxel Count');
xlabel('EdU Vox');
ylabel('Dcx Vox');

legend(datasetNames,'location','northeastoutside')

figure
pH = gca;
hold on

for i=starIdx:length(montages)
    x = (features(i).numVox(2) * features(i).voxScale)/features(i).numCC(2);
    y = (features(i).numVox(3) * features(i).voxScale)/features(i).numCC(3);
    plot(pH,x,y,...
        'MarkerFaceColor',montages(i).faceColor,...
        'marker',montages(i).marker,...
        'MarkerSize',mrkSize);
    text(x,y,num2str(i),...
        'parent',pH,...
        'color',xor([1,1,1],montages(i).faceColor),...
        'fontsize',fntSize,'fontname',font,...
        'horizontalalignment',horz,...
        'verticalalignment',vert);
end

title('Object Area');
xlabel('Avg EdU Volume \mu');
ylabel('Avg Dcx Volume \mu');

legend(datasetNames,'location','northeastoutside')

figure
pH = gca;
hold on

for i=starIdx:length(montages)
    x = features(i).numCC(2)/features(i).SVZarea;
    y = features(i).numCC(3)/features(i).SVZarea;
    plot(pH,x,y,...
        'MarkerFaceColor',montages(i).faceColor,...
        'marker',montages(i).marker,...
        'MarkerSize',mrkSize);
    text(x,y,num2str(i),...
        'parent',pH,...
        'color',xor([1,1,1],montages(i).faceColor),...
        'fontsize',fntSize,'fontname',font,...
        'horizontalalignment',horz,...
        'verticalalignment',vert);
end

title('Number of Connect Components Over the SVZ Area');
xlabel('EdU cells / SVZ Volume');
ylabel('Dcx chains / SVZ Volume');

legend(datasetNames,'location','northeastoutside')

figure
pH = gca;
hold on

for i=starIdx:length(montages)
    x = features(i).numVox(2)/features(i).SVZarea * 100;
    y = features(i).numVox(3)/features(i).SVZarea * 100;
    plot(pH,x,y,...
        'MarkerFaceColor',montages(i).faceColor,...
        'marker',montages(i).marker,...
        'MarkerSize',mrkSize);
    text(x,y,num2str(i),...
        'parent',pH,...
        'color',xor([1,1,1],montages(i).faceColor),...
        'fontsize',fntSize,'fontname',font,...
        'horizontalalignment',horz,...
        'verticalalignment',vert);
end

title('Percent of SVZ Volume');
xlabel('EdU Percent');
ylabel('Dcx Percent');

legend(datasetNames,'location','northeastoutside')
