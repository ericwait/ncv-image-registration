root = 'P:\Images\Temple\3d\SVZ\Montage\Surface\DAPI Msh1-647 VCAM-488 Olg2-514 Bcat-Cy3 GFAP-594\';
subDir = 'Contralateral Sides 7-2-15';

datasets = {'22mo-S4';
    '22mo-S5';
    '22mo-S6';
    '2mo-S4';
    '2mo-S5';
    '2mo-S6'};

subdir = 'Smoothed\normalized';
cur = 'D:\Images\Temple\Deep DAPI Mash1-647 Dcx-488 ki67-514 Laminin-Cy3 GFAP-594\2mM3';

for i=[2,4:length(datasets),1]
    imD = MicroscopeData.ReadMetadata(cur);%fullfile(root,datasets{i},subdir));
    im = MicroscopeData.Reader(imD);
    
    prgs2 = Utils.CmdlnProgress(imD.NumberOfChannels,true,sprintf('%s...',imD.DatasetName));
    for c=1:imD.NumberOfChannels
        im(:,:,:,c) = Cuda.ContrastEnhancement(im(:,:,:,c),[75,75,25],[3,3,3],1);
        im(:,:,:,c) = ImUtils.ConvertType(im(:,:,:,c),class(im),true);
        prgs2.PrintProgress(c);
    end
    prgs2.ClearProgress(true);
    
    MicroscopeData.Writer(im,fullfile(imD.imageDir,'normalized'),imD,[],[],[],true);
    
    MicroscopeData.Colors.WriteMIPcombs(im,imD,fullfile(imD.imageDir,'normalized'));
end
