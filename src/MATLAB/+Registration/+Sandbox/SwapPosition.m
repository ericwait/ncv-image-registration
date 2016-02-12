[ imageDatasets, datasetName ] = Registration.GetMontageSubMeta('P:\Images\Temple\3d\SVZ\Montage\Deep\Itga9 kd3(J4) Deep Labels 8-12-13 take2\list.txt');

for i=1:length(imageDatasets)
    imageDatasets(i).Position = Utils.SwapXY_RC(imageDatasets(i).Position);
    MicroscopeData.CreateMetadata(imageDatasets(i).imageDir,imageDatasets(i));
end
