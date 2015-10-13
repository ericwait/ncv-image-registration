function roiImage = findROI(im)
roiImage = false(size(im,1),size(im,2),size(im,3));

for c=1:size(im,4)
    curIm = im(:,:,:,c);
    curImDN = FluorescentBGRemoval(curIm);
    curImDN = CudaMex('MedianFilter',curImDN,[5,5,5]);
    
    %% interate over each column and stack
    yAxisIm = false(size(curImDN));
    for z=1:size(im,3)
        for x=1:size(im,2)
            indFirst = find(curImDN(:,x,z)>0,1,'first');
            indLast = find(curImDN(:,x,z)>0,1,'last');
            yAxisIm(indFirst:indLast,x,z) = true;
        end
    end
    %% iterate over each row and stack
    xAxisIm = false(size(curImDN));
    for z=1:size(im,3)
        for y=1:size(im,1)
            indFirst = find(curImDN(y,:,z)>0,1,'first');
            indLast = find(curImDN(y,:,z)>0,1,'last');
            xAxisIm(y,indFirst:indLast,z) = true;
        end
    end
    %% iterate over each row and column
%     zAxisIm = false(size(curImDN));
%     for x=1:size(im,2)
%         for y=1:size(im,1)
%             indFirst = find(curImDN(y,x,:)>0,1,'first');
%             indLast = find(curImDN(y,x,:)>0,1,'last');
%             zAxisIm(y,x,indFirst:indLast) = true;
%         end
%     end
    if (c==1)
        roiImage = yAxisIm & xAxisIm;% & zAxisIm;
    else
        roiImage = roiImage | (yAxisIm & xAxisIm);% & zAxisIm);
    end
end